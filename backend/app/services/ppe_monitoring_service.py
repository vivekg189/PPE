import collections
import json
import logging
import time
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger("ppe_monitoring.pipeline_service")

try:
    from app.services.detection_service import process_detections
    from app.services.model_service import get_model
    from app.services.model_registry import get_active_model_id, get_model_profile
    from app.services.ppe_association_service import PPEAssociationService
    from app.services.ppe_compliance_service import PPEComplianceService
    from app.services.worker_tracking_service import WorkerTrackingService
    from app.services.violation_timer_service import ViolationTimerService
    from app.services.alarm_service import AlarmService
    from app.services.violation_event_service import ViolationEventService
    from app.services.websocket_manager import ConnectionManager, get_websocket_manager
except ImportError:
    # Allow imports if run with backend directly on sys.path
    from services.detection_service import process_detections
    from services.model_service import get_model
    from services.model_registry import get_active_model_id, get_model_profile
    from services.ppe_association_service import PPEAssociationService
    from services.ppe_compliance_service import PPEComplianceService
    from services.worker_tracking_service import WorkerTrackingService
    from services.violation_timer_service import ViolationTimerService
    from services.alarm_service import AlarmService
    from services.violation_event_service import ViolationEventService
    from services.websocket_manager import ConnectionManager, get_websocket_manager


class PPEMonitoringService:
    """
    Consolidated real-time PPE monitoring pipeline service.
    Coordinates YOLO detection, ByteTrack tracking, worker lifecycle tracking,
    spatial PPE-to-worker association, and per-worker compliance evaluation.
    """

    def __init__(
        self,
        model=None,
        worker_tracking_service: Optional[WorkerTrackingService] = None,
        association_service: Optional[PPEAssociationService] = None,
        compliance_service: Optional[PPEComplianceService] = None,
        violation_timer_service: Optional[ViolationTimerService] = None,
        alarm_service: Optional[AlarmService] = None,
        violation_event_service: Optional[ViolationEventService] = None,
        websocket_manager: Optional[ConnectionManager] = None,
        conf_threshold: float = 0.20,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> None:
        """
        Initializes the integrated PPE monitoring pipeline with pluggable services.

        Args:
            model: Optional pre-loaded YOLO model. If None, retrieves singleton from model_service.
            worker_tracking_service: Optional WorkerTrackingService instance.
            association_service: Optional PPEAssociationService instance.
            compliance_service: Optional PPEComplianceService instance.
            violation_timer_service: Optional ViolationTimerService instance.
            alarm_service: Optional AlarmService instance.
            violation_event_service: Optional ViolationEventService instance.
            websocket_manager: Optional ConnectionManager instance for live broadcasts.
            conf_threshold: YOLO detection confidence threshold (default: 0.20).
            imgsz: Inference image size (default: 640).
            device: Inference execution device (default: 'cpu').
        """
        self._model = model
        self.conf_threshold = float(conf_threshold)
        self.imgsz = int(imgsz)
        self.device = str(device)

        # Coordinate sub-services
        self.worker_tracking_service = worker_tracking_service or WorkerTrackingService(timeout_seconds=5.0)
        self.association_service = association_service or PPEAssociationService()
        self.compliance_service = compliance_service or PPEComplianceService(min_confidence=self.conf_threshold)
        self.violation_timer_service = violation_timer_service or ViolationTimerService(default_threshold_seconds=5.0)
        self.alarm_service = alarm_service or AlarmService()
        self.violation_event_service = violation_event_service or ViolationEventService()
        self.websocket_manager = websocket_manager or get_websocket_manager()

        # Cache of the most recent evaluation result for REST API queries
        self._last_result: Optional[Dict[str, Any]] = None

        # Real-time performance tracking metrics
        self._inference_latencies: collections.deque = collections.deque(maxlen=30)
        self._frame_times: collections.deque = collections.deque(maxlen=30)
        self._latest_inference_latency_ms: float = 0.0
        self._avg_inference_latency_ms: float = 0.0
        self._max_inference_latency_ms: float = 0.0
        self._avg_frame_time_ms: float = 0.0
        self._total_detections_count: int = 0
        self._last_frame_timestamp: Optional[float] = None

    @property
    def model(self):
        """Lazy-loads YOLO model singleton on first access."""
        if self._model is None:
            self._model = get_model()
        return self._model

    def refresh_model(self) -> None:
        """Refreshes the YOLO model instance and updates compliance config to match active model."""
        self._model = get_model()
        if hasattr(self.compliance_service, "refresh_config"):
            self.compliance_service.refresh_config()
        logger.info(f"PPEMonitoringService refreshed with model: {self._model}")

    def process_frame(
        self,
        frame: Optional[np.ndarray],
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Main pipeline entrypoint: processes a single webcam frame through the full pipeline.

        Order of Operations:
            1. Validate frame input.
            2. Run YOLO inference with ByteTrack persistent tracking.
            3. Process raw detections into structured categories.
            4. Delegate to process_structured_data for pipeline evaluation.

        Args:
            frame: OpenCV BGR image array (np.ndarray) or None.
            timestamp: Optional epoch timestamp (float). Defaults to time.time().

        Returns:
            dict: Consolidated JSON-serializable monitoring payload.
        """
        ts = float(timestamp if timestamp is not None else time.time())

        # Step 1: Validate frame
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return self._empty_response(
                timestamp=ts,
                error="Invalid, empty, or unreadable video frame received.",
            )

        frame_h, frame_w = frame.shape[:2]

        # Step 2: Run YOLO detection and ByteTrack tracking
        t_infer_start = time.perf_counter()
        try:
            results = self.model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                device=self.device,
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                verbose=False,
            )
            infer_ms = (time.perf_counter() - t_infer_start) * 1000.0
            self._latest_inference_latency_ms = round(infer_ms, 2)
            self._inference_latencies.append(infer_ms)
            if self._inference_latencies:
                self._avg_inference_latency_ms = round(sum(self._inference_latencies) / len(self._inference_latencies), 2)
                self._max_inference_latency_ms = round(max(self._inference_latencies), 2)

            now_perf = time.perf_counter()
            if self._last_frame_timestamp is not None:
                dt_ms = (now_perf - self._last_frame_timestamp) * 1000.0
                self._frame_times.append(dt_ms)
                if self._frame_times:
                    self._avg_frame_time_ms = round(sum(self._frame_times) / len(self._frame_times), 2)
            self._last_frame_timestamp = now_perf
        except Exception as e:
            logger.error(f"Inference error during frame processing: {e}", exc_info=True)
            return self._empty_response(
                timestamp=ts,
                frame_width=frame_w,
                frame_height=frame_h,
                error=f"YOLO ByteTrack execution error: {str(e)}",
            )

        # Step 3: Process detection results
        model_names = getattr(self.model, "names", {})
        structured = process_detections(results, model_names)
        self._total_detections_count += structured.get("total_detections", 0)

        # Structured debug logging
        workers_list = structured.get("workers", [])
        ppe_list = structured.get("compliant_ppe", []) or structured.get("ppe", [])
        violation_list = structured.get("violations", [])
        if workers_list or ppe_list or violation_list:
            logger.info(
                f"[YOLO DETECTIONS] Total: {structured.get('total_detections', 0)} | "
                f"Workers: {len(workers_list)}, PPE: {len(ppe_list)}, Violations: {len(violation_list)}"
            )
            for worker_det in workers_list:
                logger.info(
                    f"  Worker: Person detected (Track ID: {worker_det.get('track_id')}, "
                    f"confidence: {worker_det.get('confidence', 0.0):.2f}, bbox: {worker_det.get('bbox')})"
                )
            for p in ppe_list:
                logger.info(
                    f"  PPE item: {p.get('class_name')}, confidence: {p.get('confidence', 0.0):.2f}, bbox: {p.get('bbox')}"
                )
            for v in violation_list:
                logger.info(
                    f"  Violation item: {v.get('class_name')}, confidence: {v.get('confidence', 0.0):.2f}, bbox: {v.get('bbox')}"
                )

        # Step 4-8: Complete downstream pipeline processing
        return self.process_structured_data(
            structured_detections=structured,
            frame_width=frame_w,
            frame_height=frame_h,
            timestamp=ts,
        )

    def process_structured_data(
        self,
        structured_detections: Dict[str, Any],
        frame_width: int = 1280,
        frame_height: int = 720,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Processes structured detections through lifecycle, association, and compliance.
        Enables seamless unit testing and separation of concerns.

        Pipeline Order:
            4. Update worker lifecycle records.
            5. Separate worker and PPE detections.
            6. Associate PPE detections with tracked workers.
            7. Evaluate PPE compliance for each worker.
            8. Assemble consolidated JSON-serializable result.

        Args:
            structured_detections: Categorized detections dictionary from detection_service.
            frame_width: Video frame width in pixels.
            frame_height: Video frame height in pixels.
            timestamp: Optional epoch timestamp.

        Returns:
            dict: Consolidated frame monitoring payload.
        """
        ts = float(timestamp if timestamp is not None else time.time())

        # Step 4: Update worker lifecycle records
        lifecycle_stats = self.worker_tracking_service.update(
            worker_detections=structured_detections,
            current_time=ts,
        )

        # Step 5 & 6: Separate and associate PPE detections with tracked workers
        assoc_result = self.association_service.associate(
            structured_detections=structured_detections
        )

        associations = assoc_result.get("associations", [])
        unassigned_ppe = assoc_result.get("unassigned_ppe", [])

        # Active workers in current frame (from structured detections)
        active_workers = structured_detections.get("workers", [])
        active_track_ids = [
            int(w["track_id"]) for w in active_workers if w.get("track_id") is not None
        ]

        # Step 7: Evaluate PPE compliance for each worker
        compliance_result = self.compliance_service.evaluate_all(
            associations_payload=associations,
            worker_track_ids=active_track_ids,
        )

        worker_evals = compliance_result.get("worker_evaluations", {})

        # Step 8 (part A): Assemble consolidated worker-level records
        # Map associations by worker track ID
        worker_assoc_map: Dict[int, List[Dict[str, Any]]] = {}
        for a in associations:
            wid = a.get("worker_track_id")
            if wid is not None:
                worker_assoc_map.setdefault(wid, []).append(a)

        workers_payload: List[Dict[str, Any]] = []

        for w in active_workers:
            track_id = w.get("track_id")
            if track_id is None:
                continue

            track_id = int(track_id)
            bbox = [float(v) for v in w.get("bbox", [])]
            w_eval = worker_evals.get(track_id, {})
            w_assocs = worker_assoc_map.get(track_id, [])

            overall_st = w_eval.get("overall_status", "needs_review")
            ppe_st = w_eval.get("ppe_status", {})
            missing_items = w_eval.get("missing_ppe", [])
            missing_lbls = w_eval.get("missing_ppe_labels", [])

            worker_item = {
                "worker_id": track_id,
                "worker_track_id": track_id,
                "worker_bounding_box": bbox,
                "lifecycle_status": "active",
                "overall_status": overall_st,
                "compliance_status": overall_st,
                "overall_reason": w_eval.get("overall_reason", "No evaluation data"),
                "compliant_category_count": int(w_eval.get("compliant_count", 0)),
                "violation_count": int(w_eval.get("violation_count", 0)),
                "not_detected_count": int(w_eval.get("not_detected_count", 0)),
                "uncertain_count": int(w_eval.get("uncertain_count", 0)),
                "categories": w_eval.get("categories", {}),
                "ppe_status": ppe_st,
                "missing_ppe": missing_items,
                "missing_ppe_labels": missing_lbls,
                "associated_ppe": w_assocs,
            }
            workers_payload.append(worker_item)

            logger.info(
                f"[WORKER COMPLIANCE] Worker #{track_id} -> Status: {overall_st.upper()} | "
                f"Missing PPE: {missing_lbls if missing_lbls else 'None'} | "
                f"ppe_status: {ppe_st}"
            )

        # Step 8 (part B): Update violation timers using assembled workers_payload
        # Passes the per-worker records (which contain "categories" with compliance statuses)
        # to ViolationTimerService, which starts/continues/resets timers accordingly.
        timer_summary = self.violation_timer_service.update(
            monitoring_payload={"workers": workers_payload},
            current_time=ts,
        )

        # Step 8 (part C): Update alarm service using alarm-eligible violations
        alarm_summary = self.alarm_service.update(
            alarm_eligible_violations=timer_summary.get("alarm_eligible_timers", []),
            current_time=ts,
        )

        # Step 8 (part D): Update violation events lifecycle
        event_summary = self.violation_event_service.update(
            alarm_eligible_violations=timer_summary.get("alarm_eligible_timers", []),
            alarm_summary=alarm_summary,
            current_time=ts,
        )

        # Step 9: Consolidated JSON-serializable frame result
        result = {
            "timestamp": ts,
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "detection_count": int(structured_detections.get("total_detections", 0)),
            "worker_lifecycle": lifecycle_stats,
            "total_workers": int(compliance_result.get("total_workers", len(workers_payload))),
            "total_compliant": int(compliance_result.get("compliant_workers", 0)),
            "total_non_compliant": int(compliance_result.get("non_compliant_workers", 0)),
            "total_needs_review": int(compliance_result.get("needs_review_workers", 0)),
            "workers": workers_payload,
            "unassigned_ppe": unassigned_ppe,
            "associations": associations,
            "violation_timers": timer_summary.get("active_timers", []),
            "alarm_eligible_violations": timer_summary.get("alarm_eligible_timers", []),
            "active_timer_count": int(timer_summary.get("active_timers_count", 0)),
            "alarm_eligible_count": int(timer_summary.get("alarm_eligible_count", 0)),
            "alarms": alarm_summary.get("active_alarms", []),
            "triggered_alarms": alarm_summary.get("triggered_alarms", []),
            "active_alarms_count": int(alarm_summary.get("active_alarms_count", 0)),
            "violation_events": self.violation_event_service.get_all_events(),
            "active_violation_events": self.violation_event_service.get_active_events(),
            "violation_event_summary": event_summary,
            "active_model_id": get_active_model_id(),
            "active_model_name": get_model_profile().display_name,
            "error": None,
        }
        self._last_result = result

        # Step 10: Broadcast live updates to active WebSocket clients if connected
        if self.websocket_manager is not None and self.websocket_manager.has_connections():
            self.websocket_manager.broadcast_monitoring_update(result)

        return result

    def _empty_response(
        self,
        timestamp: float,
        frame_width: int = 0,
        frame_height: int = 0,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generates a safe fallback payload on error or invalid frame."""
        return {
            "timestamp": float(timestamp),
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "detection_count": 0,
            "worker_lifecycle": self.worker_tracking_service.get_session_statistics(),
            "total_workers": 0,
            "total_compliant": 0,
            "total_non_compliant": 0,
            "total_needs_review": 0,
            "workers": [],
            "unassigned_ppe": [],
            "associations": [],
            "violation_timers": self.violation_timer_service.get_active_timers(),
            "alarm_eligible_violations": self.violation_timer_service.get_alarm_eligible_timers(),
            "active_timer_count": len(self.violation_timer_service.get_active_timers()),
            "alarm_eligible_count": len(self.violation_timer_service.get_alarm_eligible_timers()),
            "alarms": self.alarm_service.get_active_alarms(),
            "triggered_alarms": [],
            "active_alarms_count": len(self.alarm_service.get_active_alarms()),
            "violation_events": self.violation_event_service.get_all_events(),
            "active_violation_events": self.violation_event_service.get_active_events(),
            "violation_event_summary": self.violation_event_service.get_summary(current_time=timestamp),
            "error": error,
        }

    def get_latest_monitoring_data(self) -> Dict[str, Any]:
        """
        Returns the latest frame monitoring payload, or a safe baseline snapshot
        if no frame has been evaluated yet.
        """
        if self._last_result is not None:
            return self._last_result
        return self._empty_response(timestamp=time.time())

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Returns real-time AI and frame processing performance metrics."""
        fps = round(1000.0 / self._avg_frame_time_ms, 1) if self._avg_frame_time_ms > 0 else 0.0
        return {
            "latest_inference_latency_ms": self._latest_inference_latency_ms,
            "avg_inference_latency_ms": self._avg_inference_latency_ms,
            "max_inference_latency_ms": self._max_inference_latency_ms,
            "avg_frame_time_ms": self._avg_frame_time_ms,
            "fps": fps,
            "total_detections": self._total_detections_count,
            "confidence_threshold": self.conf_threshold,
            "imgsz": self.imgsz,
            "device": self.device,
        }

    def set_confidence_threshold(self, threshold: float) -> None:
        """Dynamically updates the detection confidence threshold."""
        self.conf_threshold = max(0.01, min(0.99, float(threshold)))
        if hasattr(self.compliance_service, "min_confidence"):
            self.compliance_service.min_confidence = self.conf_threshold
        logger.info(f"PPEMonitoringService confidence threshold updated to {self.conf_threshold:.2f}")


# Module-level singleton instance for shared backend / REST API access
_monitoring_service_instance: Optional[PPEMonitoringService] = None


def get_monitoring_service() -> PPEMonitoringService:
    """Returns the shared PPEMonitoringService singleton."""
    global _monitoring_service_instance
    if _monitoring_service_instance is None:
        _monitoring_service_instance = PPEMonitoringService()
    return _monitoring_service_instance


