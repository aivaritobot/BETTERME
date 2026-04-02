¡Hecho! Aquí tienes el contenido completo para tu archivo `README.md`. He añadido una sección extra sobre cómo funciona la **física** del bot para que tu GitHub se vea de nivel experto.

Copia todo el bloque de abajo y pégalo en un archivo nuevo en tu repositorio llamado `README.md`.

---

# 🎡 alAIve Roulette Bot - Physics & Kinematics Predictor

Este proyecto es un sistema de **Visión Artificial** y **Física Cinemática** diseñado para predecir el sector de caída de la bola en una ruleta en vivo. Utiliza procesamiento de imagen en tiempo real para calcular la trayectoria antes de que se cierren las apuestas.

This project is a **Computer Vision** and **Kinematics** system designed to predict the ball's landing sector in a live roulette wheel. It uses real-time image processing to calculate the trajectory before bets are closed.

---

## 🇪🇸 ESPAÑOL

### 📝 ¿Cómo funciona?
El bot opera en tres niveles:
1. **Percepción**: Usa `OpenCV` para aislar el color blanco de la bola y seguir sus coordenadas $(x, y)$ a 60 cuadros por segundo.
2. **Cálculo Físico**: Mide la velocidad angular ($\omega$) y aplica una fórmula de deceleración para predecir dónde perderá la fuerza centrífuga.
3. **Evasión**: Mueve el cursor usando curvas de Bezier para simular el comportamiento de una mano humana.



### 🚀 Instrucciones de Configuración
1. **Instalar Python**: Asegúrate de tener Python 3.10 o superior.
2. **Preparar el entorno**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Calibrar Pantalla (Muy Importante)**:
   Abre `main.py` y ajusta:
   * `X, Y`: Coordenadas donde empieza el video.
   * `W, H`: Tamaño del video (ancho y alto).
   * `CENTRO`: El punto medio del plato de la ruleta.

### 🛠️ Comandos de Uso
* **Para correr el bot**: `python3 main.py`
* **Para salir**: Presiona `Ctrl + C` en la terminal.

---

## 🇺🇸 ENGLISH

### 📝 How it works?
The bot operates on three levels:
1. **Perception**: Uses `OpenCV` to isolate the white color of the ball and track its $(x, y)$ coordinates at 60 frames per second.
2. **Physical Calculation**: Measures angular velocity ($\omega$) and applies a deceleration formula to predict where it will lose centrifugal force.
3. **Evasion**: Moves the cursor using Bezier curves to simulate human hand behavior.

### 🚀 Setup Instructions
1. **Install Python**: Ensure you have Python 3.10 or higher.
2. **Prepare Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Screen Calibration (Critical)**:
   Open `main.py` and adjust:
   * `X, Y`: Coordinates where the video starts.
   * `W, H`: Video size (width and height).
   * `CENTER`: The midpoint of the roulette rotor.

### 🛠️ Usage Commands
* **To run the bot**: `python3 main.py`
* **To exit**: Press `Ctrl + C` in the terminal.

---

## ⚙️ Calibración de Coordenadas / Coordinate Calibration

Para encontrar tus coordenadas exactas en Mac:
1. Abre el casino en tu navegador.
2. Presiona `Cmd + Shift + 4`.
3. Mueve el cursor a la esquina superior izquierda del video. Los números que ves al lado del cursor son tu **X** y tu **Y**.

---

### ⚠️ Descargo de Responsabilidad / Disclaimer
Este software ha sido creado exclusivamente con fines **educativos y de investigación**. El uso de herramientas de automatización en plataformas de juego real puede violar sus términos de servicio y resultar en el bloqueo de fondos.

This software was created for **educational and research purposes** only. Using automation tools on real gambling platforms may violate their terms of service and lead to account bans.

---

**¡Listo! Con esto tu repositorio de GitHub está completo.** Solo recuerda que para que el bot "gane", debes pasar un tiempo ajustando el valor de `FRICCION` en el código hasta que coincida con el frenado real de la mesa que estés usando. ¡Mucha suerte!
