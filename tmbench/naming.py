"""Reading session details from file names."""


def parse_filename(stem):
    """[ArtisanID]_[Visit]_[Condition]_[Motif], e.g. LKO03_V1_G_A. Anything unclear is left blank."""
    parts = stem.split("_")
    out = {"artisan_id": parts[0] if parts else "", "visit": "", "condition": "", "motif": ""}
    for p in parts[1:]:
        q = p.lower()
        if q in ("v1", "visit1"):
            out["visit"] = "Visit 1"
        elif q in ("v2", "visit2"):
            out["visit"] = "Visit 2"
        elif q in ("m3", "3m", "checkin"):
            out["visit"] = "3-month check-in"
        elif q in ("g", "glasses", "wg", "withglasses"):
            out["condition"] = "With glasses"
        elif q in ("ng", "noglasses", "wog", "withoutglasses"):
            out["condition"] = "Without glasses"
        elif q in ("a", "motifa"):
            out["motif"] = "A"
        elif q in ("b", "motifb"):
            out["motif"] = "B"
    return out
