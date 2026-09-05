# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

import sys
import logging

from faceset_curator.app_logging import configure_logging


def main() -> None:
    configure_logging()
    if "--doctor" in sys.argv:
        from faceset_curator.cli import doctor_report
        try:
            report = doctor_report("cuda")
            logging.getLogger("faceset_curator.packaging").info("Standalone doctor: %s", report)
        except Exception:
            logging.getLogger("faceset_curator.packaging").exception("Standalone doctor failed")
            raise SystemExit(1)
        raise SystemExit(0 if report["cuda_ready"] and report["inference_test"]["passed"] else 1)
    if "--gui-smoke" in sys.argv:
        try:
            from faceset_curator.gui import FaceSetCuratorApp
            app = FaceSetCuratorApp()
            app.update_idletasks()
            app.update()
            app.destroy()
            logging.getLogger("faceset_curator.packaging").info("Standalone GUI smoke passed")
        except Exception:
            logging.getLogger("faceset_curator.packaging").exception("Standalone GUI smoke failed")
            raise SystemExit(1)
        raise SystemExit(0)
    from faceset_curator.gui import main as gui_main
    gui_main()


if __name__ == "__main__":
    main()
