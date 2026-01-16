#!/usr/bin/env python3
"""
SUJBOT2 Security Notification Script

Parses Claude Code security analysis output and sends notifications
via email (Gmail SMTP) and/or MS Teams webhook based on severity.

Usage:
    echo '{"overall_status": "warning", ...}' | python security_notify.py --config config.json
    python security_notify.py --config config.json --test  # Send test notification
"""

import argparse
import json
import logging
import os
import smtplib
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict[str, Any]:
    """Load security monitoring configuration from JSON file."""
    with open(config_path) as f:
        config = json.load(f)
    return config.get("security_monitoring", {})


def get_env_value(env_var_name: str, default: str = "") -> str:
    """Get value from environment variable."""
    return os.getenv(env_var_name, default)


def format_email_html(report: dict[str, Any]) -> str:
    """Format security report as HTML email."""
    status = report.get("overall_status", "unknown")
    status_color = {
        "healthy": "#28a745",
        "warning": "#ffc107",
        "critical": "#dc3545",
    }.get(status, "#6c757d")

    # Format findings
    findings_html = ""
    for finding in report.get("findings", []):
        severity = finding.get("severity", "info")
        border_color = {
            "critical": "#dc3545",
            "warning": "#ffc107",
            "info": "#17a2b8",
        }.get(severity, "#6c757d")

        severity_badge = {
            "critical": '<span style="background:#dc3545;color:white;padding:2px 8px;border-radius:4px;font-size:12px;">CRITICAL</span>',
            "warning": '<span style="background:#ffc107;color:black;padding:2px 8px;border-radius:4px;font-size:12px;">WARNING</span>',
            "info": '<span style="background:#17a2b8;color:white;padding:2px 8px;border-radius:4px;font-size:12px;">INFO</span>',
        }.get(severity, severity.upper())

        recommendation_html = ""
        if finding.get("recommendation"):
            recommendation_html = f'<div style="color:#007bff;margin-top:8px;"><strong>Doporuceni:</strong> {finding.get("recommendation")}</div>'

        findings_html += f"""
        <div style="border-left:4px solid {border_color};padding:12px;margin:12px 0;background:#f8f9fa;border-radius:0 4px 4px 0;">
            <div style="margin-bottom:8px;">{severity_badge} <strong>{finding.get('title', 'Untitled')}</strong></div>
            <div style="color:#555;line-height:1.5;">{finding.get('description', '')}</div>
            {recommendation_html}
        </div>
        """

    # Format metrics
    metrics = report.get("metrics", {})
    metrics_rows = ""
    metric_labels = {
        "containers_healthy": "Kontejnery OK",
        "containers_unhealthy": "Kontejnery s problemem",
        "containers_total": "Kontejnery celkem",
        "error_count_period": "Chyby (perioda)",
        "error_count_24h": "Chyby (24h)",
        "warning_count_period": "Varovani (perioda)",
        "disk_usage_percent": "Disk (%)",
        "memory_usage_percent": "Pamet (%)",
        "failed_logins_period": "Neuspesna prihlaseni",
        "failed_logins_24h": "Neuspesna prihlaseni (24h)",
        "ssl_days_remaining": "SSL platnost (dni)",
    }

    row_bg = True
    for key, label in metric_labels.items():
        if key in metrics:
            value = metrics[key]
            bg_style = 'background:#f0f0f0;' if row_bg else ''
            metrics_rows += f"""
            <tr style="{bg_style}">
                <td style="padding:8px;border:1px solid #ddd;">{label}</td>
                <td style="padding:8px;border:1px solid #ddd;text-align:right;">{value}</td>
            </tr>
            """
            row_bg = not row_bg

    metrics_html = f"""
    <table style="width:100%;border-collapse:collapse;margin:12px 0;">
        {metrics_rows}
    </table>
    """ if metrics_rows else "<p style='color:#6c757d;'>Zadne metriky nejsou k dispozici.</p>"

    # Format highlights (for daily reports)
    highlights_html = ""
    if report.get("highlights"):
        highlights_items = "".join(f"<li>{h}</li>" for h in report["highlights"])
        highlights_html = f"""
        <h2 style="color:#333;margin-top:20px;">Hlavni body</h2>
        <ul style="line-height:1.8;">{highlights_items}</ul>
        """

    # Format action items (for daily reports)
    actions_html = ""
    if report.get("action_items"):
        actions_items = ""
        for action in report["action_items"]:
            priority_color = {
                "high": "#dc3545",
                "medium": "#ffc107",
                "low": "#17a2b8",
            }.get(action.get("priority", "low"), "#6c757d")
            actions_items += f"""
            <li style="margin-bottom:8px;">
                <span style="color:{priority_color};font-weight:bold;">[{action.get('priority', 'low').upper()}]</span>
                {action.get('action', '')}
                {f" (do: {action.get('due', '')})" if action.get('due') else ''}
            </li>
            """
        actions_html = f"""
        <h2 style="color:#333;margin-top:20px;">Akce k provedeni</h2>
        <ul style="line-height:1.8;">{actions_items}</ul>
        """

    report_type = report.get("report_type", "regular_check")
    report_title = "Denni souhrn" if report_type == "daily_summary" else "Bezpecnostni report"
    period = report.get("period", "4h")

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="font-family:Arial,Helvetica,sans-serif;max-width:800px;margin:0 auto;padding:20px;background:#ffffff;">
        <div style="background:{status_color};color:white;padding:20px;border-radius:8px 8px 0 0;">
            <h1 style="margin:0;font-size:24px;">SUJBOT2 {report_title}</h1>
            <p style="margin:8px 0 0 0;font-size:16px;">Status: <strong>{status.upper()}</strong> | Perioda: {period}</p>
        </div>

        <div style="border:1px solid #ddd;border-top:none;padding:20px;border-radius:0 0 8px 8px;">
            <h2 style="color:#333;margin-top:0;">Souhrn</h2>
            <p style="font-size:16px;line-height:1.6;">{report.get('summary', 'Zadny souhrn neni k dispozici')}</p>

            {highlights_html}

            <h2 style="color:#333;margin-top:20px;">Nalezy</h2>
            {findings_html if findings_html else '<p style="color:#28a745;">Zadne problemy nebyly nalezeny.</p>'}

            <h2 style="color:#333;margin-top:20px;">Metriky</h2>
            {metrics_html}

            {actions_html}

            <hr style="border:none;border-top:1px solid #ddd;margin:24px 0;">
            <p style="color:#999;font-size:12px;">
                Vygenerovano: {report.get('timestamp', datetime.utcnow().isoformat())}Z<br>
                Server: sujbot.fjfi.cvut.cz<br>
                <em>Tento report byl automaticky vygenerovan systemem SUJBOT2 Security Monitor.</em>
            </p>
        </div>
    </body>
    </html>
    """
    return html


def format_email_text(report: dict[str, Any]) -> str:
    """Format security report as plain text email."""
    lines = [
        "=" * 60,
        "SUJBOT2 SECURITY REPORT",
        "=" * 60,
        "",
        f"Status: {report.get('overall_status', 'unknown').upper()}",
        f"Cas: {report.get('timestamp', 'N/A')}",
        "",
        "SOUHRN",
        "-" * 40,
        report.get("summary", "Zadny souhrn"),
        "",
    ]

    if report.get("findings"):
        lines.extend(["NALEZY", "-" * 40])
        for finding in report["findings"]:
            lines.append(
                f"[{finding.get('severity', 'info').upper()}] {finding.get('title', 'Untitled')}"
            )
            lines.append(f"  {finding.get('description', '')}")
            if finding.get("recommendation"):
                lines.append(f"  -> {finding.get('recommendation')}")
            lines.append("")

    if report.get("metrics"):
        lines.extend(["METRIKY", "-" * 40])
        for key, value in report["metrics"].items():
            lines.append(f"  {key}: {value}")

    lines.extend(["", "=" * 60])
    return "\n".join(lines)


def send_email(config: dict[str, Any], report: dict[str, Any], severity: str) -> bool:
    """Send email notification via Gmail SMTP."""
    email_config = config.get("notifications", {}).get("email", {})

    if not email_config.get("enabled", False):
        logger.info("Email notifications disabled in config")
        return False

    # Check severity filter
    email_on = config.get("notifications", {}).get("severity_filter", {}).get(
        "email_on", ["critical", "warning"]
    )
    if severity not in email_on and severity != "test":
        logger.info(f"Severity '{severity}' not in email filter {email_on}, skipping")
        return False

    try:
        smtp_host = email_config.get("smtp_host", "smtp.gmail.com")
        smtp_port = email_config.get("smtp_port", 587)
        smtp_user = get_env_value(
            email_config.get("smtp_user_env", "GMAIL_USER")
        )
        smtp_password = get_env_value(
            email_config.get("smtp_password_env", "GMAIL_APP_PASSWORD")
        )
        from_address = email_config.get("from_address", smtp_user)
        recipients = email_config.get("recipients", [])
        subject_prefix = email_config.get("subject_prefix", "[SUJBOT Monitor]")

        if not all([smtp_host, smtp_user, smtp_password, recipients]):
            logger.warning("Email configuration incomplete:")
            logger.warning(f"  smtp_host: {'set' if smtp_host else 'MISSING'}")
            logger.warning(f"  smtp_user: {'set' if smtp_user else 'MISSING'}")
            logger.warning(f"  smtp_password: {'set' if smtp_password else 'MISSING'}")
            logger.warning(f"  recipients: {recipients if recipients else 'MISSING'}")
            return False

        # Build email
        msg = MIMEMultipart("alternative")

        status_text = report.get("overall_status", "unknown").upper()
        summary = report.get("summary", "Security Report")
        subject = f"{subject_prefix} {status_text}: {summary[:50]}"

        msg["Subject"] = subject
        msg["From"] = from_address
        msg["To"] = ", ".join(recipients)

        # Attach both plain text and HTML versions
        text_content = format_email_text(report)
        html_content = format_email_html(report)

        msg.attach(MIMEText(text_content, "plain", "utf-8"))
        msg.attach(MIMEText(html_content, "html", "utf-8"))

        # Send via SMTP with TLS
        logger.info(f"Connecting to {smtp_host}:{smtp_port}...")
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(from_address, recipients, msg.as_string())

        logger.info(f"Email sent successfully to {recipients}")
        return True

    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTP authentication failed: {e}")
        logger.error("Hint: For Gmail, use App Password from https://myaccount.google.com/apppasswords")
        return False
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False


def _build_teams_payload(report: dict[str, Any], channel_name: str | None = None) -> dict[str, Any]:
    """Build MessageCard payload for Teams webhook."""
    status = report.get("overall_status", "unknown")
    theme_color = {
        "healthy": "28a745",
        "warning": "ffc107",
        "critical": "dc3545",
    }.get(status, "6c757d")

    status_emoji = {
        "healthy": "✅",
        "warning": "⚠️",
        "critical": "🚨",
    }.get(status, "❓")

    # Build facts from metrics
    facts = [
        {"name": "Status", "value": f"{status_emoji} {status.upper()}"},
        {"name": "Cas", "value": report.get("timestamp", "N/A")},
    ]

    metrics = report.get("metrics", {})
    if "containers_healthy" in metrics:
        facts.append({
            "name": "Kontejnery",
            "value": f"{metrics.get('containers_healthy', '?')}/{metrics.get('containers_total', '?')} OK"
        })
    if "error_count_period" in metrics or "error_count_24h" in metrics:
        error_count = metrics.get("error_count_24h", metrics.get("error_count_period", "?"))
        facts.append({"name": "Chyby", "value": str(error_count)})
    if "ssl_days_remaining" in metrics:
        facts.append({"name": "SSL platnost", "value": f"{metrics['ssl_days_remaining']} dni"})

    # Build critical findings section
    findings_text = ""
    critical_findings = [f for f in report.get("findings", []) if f.get("severity") == "critical"]
    if critical_findings:
        findings_text = "\n\n**Kriticke nalezy:**\n"
        for f in critical_findings[:3]:  # Max 3 findings
            findings_text += f"- {f.get('title', 'Untitled')}: {f.get('description', '')[:100]}\n"

    # MessageCard format (O365 Connector / Power Automate Workflow)
    payload = {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": theme_color,
        "summary": f"SUJBOT Security: {status.upper()}",
        "sections": [
            {
                "activityTitle": f"{status_emoji} SUJBOT2 Security Alert",
                "activitySubtitle": report.get("summary", "Security monitoring report"),
                "facts": facts,
                "markdown": True,
                "text": findings_text if findings_text else None,
            }
        ],
    }

    # Remove None values
    payload["sections"][0] = {k: v for k, v in payload["sections"][0].items() if v is not None}

    return payload


def _send_to_webhook(webhook_url: str, report: dict[str, Any], channel_name: str | None = None) -> bool:
    """Send notification to a single Teams webhook."""
    try:
        payload = _build_teams_payload(report, channel_name)

        logger.info(f"Sending Teams notification to channel '{channel_name or 'default'}'...")
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()

        logger.info(f"Teams notification sent successfully to '{channel_name or 'default'}'")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send Teams notification to '{channel_name or 'default'}': {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending Teams notification: {e}")
        return False


def send_teams(config: dict[str, Any], report: dict[str, Any], severity: str) -> bool:
    """Send MS Teams notification via webhook(s).

    Supports multi-channel configuration where different severity levels
    can be routed to different Teams channels.
    """
    teams_config = config.get("notifications", {}).get("teams", {})

    if not teams_config.get("enabled", False):
        logger.info("Teams notifications disabled in config")
        return False

    # Support for multi-channel configuration
    channels = teams_config.get("channels", [])

    # Fallback to legacy single-webhook configuration
    if not channels:
        legacy_webhook_env = teams_config.get("webhook_url_env", "TEAMS_WEBHOOK_URL")
        legacy_severity = config.get("notifications", {}).get("severity_filter", {}).get(
            "teams_on", ["critical"]
        )
        channels = [{
            "name": "default",
            "webhook_url_env": legacy_webhook_env,
            "severity": legacy_severity,
        }]

    success_count = 0
    for channel in channels:
        channel_name = channel.get("name", "unknown")
        channel_severity = channel.get("severity", ["critical"])

        # Check if this channel should receive this severity level
        if severity not in channel_severity and severity != "test":
            logger.debug(f"Severity '{severity}' not in channel '{channel_name}' filter {channel_severity}")
            continue

        webhook_url = get_env_value(channel.get("webhook_url_env", ""))
        if not webhook_url:
            logger.warning(f"Teams webhook URL not configured for channel '{channel_name}' "
                          f"(env: {channel.get('webhook_url_env', 'N/A')})")
            continue

        if _send_to_webhook(webhook_url, report, channel_name):
            success_count += 1

    if success_count > 0:
        logger.info(f"Teams notifications sent to {success_count} channel(s)")
    else:
        logger.info("No Teams notifications sent (filtered by severity or not configured)")

    return success_count > 0


def main():
    parser = argparse.ArgumentParser(
        description="SUJBOT2 Security Notification Script"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config.json",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Send test notification (ignores severity filter)",
    )
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Handle test mode
    if args.test:
        logger.info("Test mode - sending test notifications")
        test_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "summary": "Testovaci notifikace - system funguje spravne",
            "findings": [
                {
                    "category": "test",
                    "severity": "info",
                    "title": "Test Notification",
                    "description": "Toto je testovaci zprava pro overeni funkcnosti notifikaci",
                    "recommendation": "Zadna akce neni potreba",
                }
            ],
            "metrics": {
                "containers_healthy": 6,
                "containers_total": 6,
                "ssl_days_remaining": 45,
            },
        }
        email_sent = send_email(config, test_report, "test")
        teams_sent = send_teams(config, test_report, "test")
        logger.info(f"Test results: email={'sent' if email_sent else 'failed'}, teams={'sent' if teams_sent else 'failed'}")
        sys.exit(0 if (email_sent or teams_sent) else 1)

    # Read report from stdin
    try:
        report_json = sys.stdin.read()
        if not report_json.strip():
            logger.error("No input received on stdin")
            sys.exit(1)
        report = json.loads(report_json)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse report JSON: {e}")
        logger.error(f"Input was: {report_json[:500]}...")
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "warning",
            "summary": "Nepodarilo se parsovat vystup z analyzy",
            "findings": [
                {
                    "category": "internal",
                    "severity": "warning",
                    "title": "Parse Error",
                    "description": f"JSON parse error: {e}",
                    "recommendation": "Zkontrolujte format vystupu z Claude Code",
                }
            ],
            "metrics": {},
        }

    severity = report.get("overall_status", "unknown")
    logger.info(f"Processing report: {severity.upper()} - {report.get('summary', 'No summary')[:50]}")

    # Send notifications
    email_sent = send_email(config, report, severity)
    teams_sent = send_teams(config, report, severity)

    if email_sent or teams_sent:
        logger.info("Notifications sent successfully")
    else:
        logger.info("No notifications sent (filtered by severity or disabled)")

    # Exit with appropriate code based on severity
    if severity == "critical":
        sys.exit(2)
    elif severity == "warning":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
