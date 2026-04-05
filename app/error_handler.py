from __future__ import annotations


def to_user_error(exc: Exception) -> tuple[str, str]:
    msg = str(exc).strip().lower()
    if "libgl.so.1" in msg:
        return (
            "Falta una librería gráfica del sistema.",
            "Instala paquetes OpenGL del sistema y reinicia la app.",
        )
    if "no module named" in msg:
        return (
            "Faltan dependencias de Python.",
            "Ejecuta el instalador del proyecto o reinstala dependencias.",
        )
    if "permission" in msg:
        return (
            "La app no tiene permisos suficientes.",
            "Revisa permisos de cámara/pantalla y vuelve a intentar.",
        )
    return (
        "No se pudo continuar con la sesión.",
        "Pulsa Reset y revisa la sección Avanzado para más detalles.",
    )
