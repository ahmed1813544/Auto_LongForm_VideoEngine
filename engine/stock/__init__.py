"""Stock package: Clip descriptor + provider adapters + factory."""
from engine.stock.base import (DEFAULT_MAX_H, DEFAULT_SIZE_CAP, Clip,
                               StockProvider, search_all)
from engine.stock.pexels import Pexels
from engine.stock.pixabay import Pixabay


def providers_for(cfg, *, wanted=("pixabay", "pexels"), log=None):
    """Instantiate only providers whose API key is configured."""
    ps = []
    for name in wanted:
        if name == "pixabay":
            if cfg.pixabay_key:
                ps.append(Pixabay(cfg.pixabay_key, log=log))
            elif log:
                log.warn("PIXABAY_API_KEY not set — Pixabay skipped")
        elif name == "pexels":
            if cfg.pexels_key:
                ps.append(Pexels(cfg.pexels_key, log=log))
            elif log:
                log.warn("PEXELS_API_KEY not set — Pexels skipped")
    return ps


__all__ = ["Clip", "StockProvider", "search_all", "Pexels", "Pixabay",
           "providers_for", "DEFAULT_MAX_H", "DEFAULT_SIZE_CAP"]
