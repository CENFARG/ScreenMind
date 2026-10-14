"""
Sensitive Data Filter
Detects and redacts credit cards, SSNs, API keys, passwords, emails and
Argentine PII (CUIT/CUIL, CBU/CVU, bank aliases, DNI) from OCR text before
it is stored in the database or passed to AI models.

Fail-closed contract: use redact_or_raise() ahead of persistence/inference.
If redaction cannot be guaranteed it raises FilterFailure, and the caller MUST
quarantine the text locally instead of persisting it or sending it to a model.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger("screenmind.privacy.data_filter")


# ── Check-digit constants ────────────────────────────────────────────

# AFIP CUIT/CUIL valid prefixes (20/23/24/27 physical, 30/33/34 legal).
_CUIT_PREFIXES = ("20", "23", "24", "27", "30", "33", "34")
# BCRA CBU/CVU weights: block 1 (bank+branch, 7 digits), block 2 (account, 13).
_CBU_W1 = (7, 1, 3, 9, 7, 1, 3)
_CBU_W2 = (3, 9, 7, 1, 3, 9, 7, 1, 3, 9, 7, 1, 3)


# ── Validators ───────────────────────────────────────────────────────

def _luhn_check(number: str) -> bool:
    """Validate a credit card number using the Luhn algorithm."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _cuit_check(number: str) -> bool:
    """Validate an Argentine CUIT/CUIL via the AFIP mod-11 check digit.

    Format PPDDDDDDDDC (11 digits). The check digit is derived from the first
    10 digits weighted by (5,4,3,2,7,6,5,4,3,2); remainder 0 -> 0, 1 -> 9,
    otherwise 11 - remainder.
    """
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) != 11:
        return False
    if "".join(str(d) for d in digits[:2]) not in _CUIT_PREFIXES:
        return False
    total = sum(d * w for d, w in zip(digits[:10], (5, 4, 3, 2, 7, 6, 5, 4, 3, 2)))
    rest = total % 11
    check = 0 if rest == 0 else (9 if rest == 1 else 11 - rest)
    return check == digits[10]


def _cbu_check(number: str) -> bool:
    """Validate an Argentine CBU/CVU (22 digits) via the BCRA mod-10 check.

    Block 1 = 8 digits (3 bank + 4 branch + 1 check), block 2 = 14 digits
    (13 account + 1 check). Both check digits use a weighted mod-10 scheme,
    so a valid CVU (same 22-digit layout) passes the identical test.
    """
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) != 22:
        return False
    b1, b2 = digits[:8], digits[8:]
    dv1 = (10 - sum(d * w for d, w in zip(b1[:7], _CBU_W1)) % 10) % 10
    dv2 = (10 - sum(d * w for d, w in zip(b2[:13], _CBU_W2)) % 10) % 10
    return dv1 == b1[7] and dv2 == b2[13]


# ── Pattern Definitions ──────────────────────────────────────────────

PATTERNS = {
    "credit_card": {
        "label": "Credit Card",
        "regex": re.compile(
            r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"
        ),
        "replacement": "[REDACTED:card]",
        "validator": _luhn_check,
    },
    "ssn": {
        "label": "SSN",
        "regex": re.compile(
            r"\b\d{3}-\d{2}-\d{4}\b"
        ),
        "replacement": "[REDACTED:ssn]",
    },
    "api_key": {
        "label": "API Key",
        "regex": re.compile(
            r"\b(?:"
            r"sk-ant-[A-Za-z0-9\-]{20,}"    # Anthropic (before sk- to prevent false match)
            r"|sk-[A-Za-z0-9]{20,}"          # OpenAI
            r"|sk_live_[A-Za-z0-9]{20,}"     # Stripe live
            r"|sk_test_[A-Za-z0-9]{20,}"     # Stripe test
            r"|pk_live_[A-Za-z0-9]{20,}"     # Stripe publishable
            r"|ghp_[A-Za-z0-9]{36,}"         # GitHub PAT
            r"|github_pat_[A-Za-z0-9_]{20,}" # GitHub fine-grained PAT
            r"|gho_[A-Za-z0-9]{36,}"         # GitHub OAuth
            r"|AKIA[A-Z0-9]{16}"             # AWS Access Key
            r"|xox[bps]-[A-Za-z0-9\-]{10,}"  # Slack
            r"|glpat-[A-Za-z0-9\-]{20,}"     # GitLab
            r"|AIza[A-Za-z0-9\-_]{35}"       # Google API
            r"|whsec_[A-Za-z0-9]{20,}"       # Webhook secrets
            r"|ntn_[A-Za-z0-9]{20,}"         # Notion
            r")\b",
            re.ASCII,
        ),
        "replacement": "[REDACTED:key]",
    },
    "jwt": {
        "label": "JWT Token",
        "regex": re.compile(
            r"\beyJ[A-Za-z0-9_-]{20,}\.eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\b"
        ),
        "replacement": "[REDACTED:jwt]",
    },
    "password": {
        "label": "Password",
        "regex": re.compile(
            r"(?i)(?:password|passwd|pwd|secret|token|api.?key)"
            r"\s*[:=]\s*"
            r"[\"']?(\S{4,})[\"']?",
        ),
        "replacement": r"[REDACTED:password]",
    },
    "email": {
        "label": "Email Address",
        "regex": re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"
        ),
        "replacement": "[REDACTED:email]",
    },
    # ── Argentina (Ley 25.326) ───────────────────────────────────────
    "cuit": {
        "label": "CUIT/CUIL",
        # 11 digits, optionally hyphenated: 20-11111111-2
        "regex": re.compile(
            r"\b\d{2}-?\d{8}-?\d{1}\b"
        ),
        "replacement": "[REDACTED:cuit]",
        "validator": _cuit_check,
    },
    "cbu": {
        "label": "CBU/CVU",
        # 22 consecutive digits (CBU and CVU share this layout)
        "regex": re.compile(
            r"\b\d{22}\b"
        ),
        "replacement": "[REDACTED:cbu]",
        "validator": _cbu_check,
    },
    "alias": {
        "label": "Bank alias (CBU/CVU)",
        # Contextual: keyword alias/cbu/cvu followed by a dotted token
        "regex": re.compile(
            r"(?i)\b(?:alias|cbu|cvu)\b[\s:]{0,3}([a-z0-9]+(?:\.[a-z0-9]+){1,4})\b"
        ),
        "replacement": "[REDACTED:alias]",
    },
    "dni": {
        "label": "DNI",
        # 7-8 digits only when anchored to a dni/documento context keyword
        "regex": re.compile(
            r"(?i)\b(?:dni|documento|doc)\b\.?[\s:\-]{0,4}"
            r"(\d{7,8}|\d{1,3}(?:\.\d{3}){2})\b"
        ),
        "replacement": "[REDACTED:dni]",
    },
}


# Types enabled when the caller does not specify any.
DEFAULT_ENABLED_TYPES = [
    "credit_card", "ssn", "api_key", "jwt", "password", "email",
    "cuit", "cbu", "alias", "dni",
]


class FilterFailure(RuntimeError):
    """Redaction could not be guaranteed — the caller must quarantine the text.

    The message intentionally carries no matched value, so it is safe to log.
    """


def filter_sensitive_text(
    text: str,
    enabled_types: Optional[list] = None,
) -> dict:
    """
    Scan text and redact sensitive data.

    Args:
        text: Raw OCR / organized text
        enabled_types: List of pattern keys to apply.
            Defaults to DEFAULT_ENABLED_TYPES.

    Returns:
        {
            "clean_text": "...",
            "redacted_count": 3,
            "types_found": ["credit_card", "api_key"],
            "details": [
                {"type": "credit_card", "count": 2},
                {"type": "api_key", "count": 1},
            ]
        }
    """
    if not text:
        return {
            "clean_text": text or "",
            "redacted_count": 0,
            "types_found": [],
            "details": [],
        }

    if enabled_types is None:
        enabled_types = list(DEFAULT_ENABLED_TYPES)

    clean = text
    total_redacted = 0
    types_found = []
    details = []

    for ptype in enabled_types:
        pattern_info = PATTERNS.get(ptype)
        if not pattern_info:
            continue

        regex = pattern_info["regex"]
        replacement = pattern_info["replacement"]

        # Count matches before replacing
        matches = regex.findall(clean)
        count = len(matches)

        if count > 0:
            # If pattern has a validator (e.g. Luhn for credit cards, mod-11
            # for CUIT, mod-10 for CBU), only redact matches that pass it.
            validator = pattern_info.get("validator")
            if validator:
                def _validated_sub(m):
                    return replacement if validator(m.group()) else m.group()
                before_count = clean.count(replacement)
                clean = regex.sub(_validated_sub, clean)
                count = clean.count(replacement) - before_count
            else:
                clean = regex.sub(replacement, clean)

            if count > 0:
                total_redacted += count
                types_found.append(ptype)
                details.append({"type": ptype, "count": count})

    if total_redacted > 0:
        logger.info(f"Redacted {total_redacted} sensitive item(s): {', '.join(types_found)}")

    return {
        "clean_text": clean,
        "redacted_count": total_redacted,
        "types_found": types_found,
        "details": details,
    }


def redact_or_raise(
    text: str,
    enabled_types: Optional[list] = None,
) -> dict:
    """Fail-closed wrapper: redact, then verify the redaction actually held.

    Runs filter_sensitive_text and re-scans the cleaned text for every enabled
    pattern. If any still-valid match survives (or the redactor itself raised),
    FilterFailure is raised so the caller quarantines the text instead of
    persisting it or sending it to inference.

    The error message never contains a matched value.
    """
    types = list(enabled_types) if enabled_types else list(DEFAULT_ENABLED_TYPES)

    try:
        result = filter_sensitive_text(text, types)
    except Exception as exc:  # noqa: BLE001 — any failure must fail closed
        raise FilterFailure(f"redactor raised ({type(exc).__name__})") from exc

    clean = result.get("clean_text") or ""
    for ptype in types:
        info = PATTERNS.get(ptype)
        if not info:
            continue
        validator = info.get("validator")
        for match in info["regex"].finditer(clean):
            if validator is None or validator(match.group()):
                raise FilterFailure(f"redaction invariant violated for '{ptype}'")

    return result


def parse_enabled_types(types_str: str) -> list:
    """Parse comma-separated filter types string into a list."""
    if not types_str:
        return list(DEFAULT_ENABLED_TYPES)
    return [t.strip() for t in types_str.split(",") if t.strip() in PATTERNS]
