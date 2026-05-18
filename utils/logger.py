# utils/logger.py
# ============================================================
# Logger central con colores y secciones visuales.
# Usado por todos los módulos del pipeline.
# ============================================================

import sys
from datetime import datetime

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLOR = True
except ImportError:
    COLOR = False


class Logger:
    """
    Logger con niveles de color y secciones visuales.
    Imprime en consola con timestamps opcionales.
    """

    ANCHO = 60

    @staticmethod
    def _color(texto, color_code):
        if COLOR:
            return color_code + texto + Style.RESET_ALL
        return texto

    @staticmethod
    def _ts():
        return datetime.now().strftime("%H:%M:%S")

    # ── Niveles ───────────────────────────────────────────────
    @staticmethod
    def info(msg):
        ts = Logger._ts()
        print(Logger._color(f"[{ts}] ℹ  {msg}", Fore.CYAN if COLOR else ""))

    @staticmethod
    def ok(msg):
        ts = Logger._ts()
        print(Logger._color(f"[{ts}] ✔  {msg}", Fore.GREEN if COLOR else ""))

    @staticmethod
    def warn(msg):
        ts = Logger._ts()
        print(Logger._color(f"[{ts}] ⚠  {msg}", Fore.YELLOW if COLOR else ""))

    @staticmethod
    def error(msg):
        ts = Logger._ts()
        print(Logger._color(f"[{ts}] ✘  {msg}", Fore.RED if COLOR else ""), file=sys.stderr)

    @staticmethod
    def result(msg):
        ts = Logger._ts()
        print(Logger._color(f"[{ts}] ►  {msg}", Fore.MAGENTA if COLOR else ""))

    # ── Secciones visuales ────────────────────────────────────
    @staticmethod
    def seccion(titulo):
        linea = "═" * Logger.ANCHO
        print()
        print(Logger._color(linea, Fore.BLUE if COLOR else ""))
        print(Logger._color(f"  {titulo.upper()}", Fore.BLUE if COLOR else ""))
        print(Logger._color(linea, Fore.BLUE if COLOR else ""))

    @staticmethod
    def subseccion(titulo):
        linea = "─" * Logger.ANCHO
        print()
        print(Logger._color(linea, Fore.CYAN if COLOR else ""))
        print(Logger._color(f"  {titulo}", Fore.CYAN if COLOR else ""))
        print(Logger._color(linea, Fore.CYAN if COLOR else ""))

    @staticmethod
    def separador():
        print(Logger._color("·" * Logger.ANCHO, Fore.WHITE if COLOR else ""))

    @staticmethod
    def preguntar(msg):
        """Pregunta al usuario y devuelve la respuesta en minúsculas."""
        respuesta = input(Logger._color(f"\n❓  {msg} ", Fore.YELLOW if COLOR else ""))
        return respuesta.strip().lower()
