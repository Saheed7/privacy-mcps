"""Alert Microservice (AMS) — Layer 5.
Monitors model outputs, classifies threats, dispatches notifications."""

import logging
import json
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)


# Severity mapping for multi-class attack categories
SEVERITY_MAP = {
    "tcp-syn": "critical", "udp-flood": "critical", "http-flood": "critical",
    "icmp-flood": "critical", "dos": "critical", "ddos": "critical",
    "arp-spoof": "high", "dns-spoof": "high", "dns-poison": "high",
    "mitm": "high", "man-in-the-middle": "high",
    "sql-injection": "high", "xss": "high", "cmd-injection": "high",
    "injection": "high",
    "port-scan": "medium", "os-fingerprint": "medium", "vuln-scan": "medium",
    "ping-sweep": "medium", "reconnaissance": "medium",
    "ransomware": "critical", "backdoor": "critical", "malware": "critical",
    "brute-force": "medium", "ble-spoof": "medium",
    "mqtt-exploit": "high", "coap-flood": "high",
}


class AMS:
    """Alert Microservice for real-time threat notification."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.alert_log = []

    def classify_threat(self, prediction: Dict) -> Dict:
        """
        Classify detected threat and assign severity.

        Args:
            prediction: Dict with 'predicted_class', 'label', 'confidence', 'source_ip', etc.

        Returns:
            Alert dict with severity, recommended actions.
        """
        label = prediction.get("label", "unknown").lower().strip()

        # Binary: any non-benign is a threat
        if label in ["benign", "normal", "0"]:
            return None  # No alert for benign traffic

        # Determine severity
        severity = "low"
        for key, sev in SEVERITY_MAP.items():
            if key in label:
                severity = sev
                break

        alert = {
            "alert_id": f"AMS-{len(self.alert_log)+1:06d}",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "attack_type": prediction.get("label", "Unknown"),
            "predicted_class": prediction.get("predicted_class"),
            "confidence": prediction.get("confidence", 0.0),
            "severity": severity,
            "source_ip": prediction.get("source_ip", "N/A"),
            "target_device": prediction.get("target_device", "N/A"),
            "recommended_actions": self._get_actions(severity),
        }

        self.alert_log.append(alert)
        logger.warning(f"[AMS] ALERT: {alert['severity'].upper()} — "
                       f"{alert['attack_type']} (conf: {alert['confidence']:.3f})")
        return alert

    def _get_actions(self, severity: str) -> List[str]:
        """Get recommended mitigation actions based on severity."""
        actions = {
            "critical": [
                "Immediately quarantine affected IoMT device",
                "Block source IP at network firewall",
                "Activate backup communication channel",
                "Escalate to SOC duty manager",
                "Initiate incident response procedure",
            ],
            "high": [
                "Isolate affected network segment",
                "Block source IP at edge firewall",
                "Notify healthcare IT administrator",
                "Log for forensic analysis",
            ],
            "medium": [
                "Monitor source IP for continued activity",
                "Log event for trend analysis",
                "Notify security operations team",
            ],
            "low": [
                "Log event for audit trail",
                "Include in periodic security report",
            ],
        }
        return actions.get(severity, actions["low"])

    def dispatch_admin_alert(self, alert: Dict):
        """(i) Send alert to healthcare IT administrators."""
        logger.info(f"[AMS] Admin alert dispatched: {alert['alert_id']}")
        # In production: send email/push notification via SMTP/webhook

    def dispatch_soc_alert(self, alert: Dict):
        """(ii) Send CEF/Syslog formatted alert to SIEM."""
        cef = (f"CEF:0|PrivacyMCPS|AMS|1.0|{alert['attack_type']}|"
               f"Intrusion Detected|{self._severity_to_int(alert['severity'])}|"
               f"src={alert['source_ip']} dst={alert['target_device']} "
               f"confidence={alert['confidence']:.3f}")
        logger.info(f"[AMS] SOC/SIEM alert: {cef}")
        # In production: send to syslog endpoint

    def dispatch_auto_response(self, alert: Dict):
        """(iii) Trigger automated incident response."""
        if alert["severity"] in ["critical", "high"]:
            logger.info(f"[AMS] Auto-response triggered for {alert['alert_id']}: "
                        f"quarantine + block")
            # In production: call firewall API, device management API

    def log_event(self, alert: Dict, filepath: str = "results/alert_log.jsonl"):
        """(iv) Comprehensive event logging."""
        with open(filepath, "a") as f:
            f.write(json.dumps(alert) + "\n")

    def process_prediction(self, prediction: Dict):
        """Full alert pipeline: classify → dispatch all channels."""
        alert = self.classify_threat(prediction)
        if alert is None:
            return  # Benign, no alert

        self.dispatch_admin_alert(alert)
        self.dispatch_soc_alert(alert)
        self.dispatch_auto_response(alert)
        self.log_event(alert)
        return alert

    @staticmethod
    def _severity_to_int(severity: str) -> int:
        return {"critical": 10, "high": 7, "medium": 4, "low": 1}.get(severity, 0)

    def get_statistics(self) -> Dict:
        """Return alert statistics summary."""
        if not self.alert_log:
            return {"total_alerts": 0}
        severities = [a["severity"] for a in self.alert_log]
        return {
            "total_alerts": len(self.alert_log),
            "critical": severities.count("critical"),
            "high": severities.count("high"),
            "medium": severities.count("medium"),
            "low": severities.count("low"),
        }
