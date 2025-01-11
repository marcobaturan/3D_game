[ES]

# MazeVisionGame

## Descripción

MazeVisionGame es un juego de laberinto en 3D que utiliza visión por computadora para analizar el entorno y tomar decisiones de navegación. El juego incorpora un modelo de lenguaje grande (LLM) para decidir las acciones a tomar basándose en la descripción del entorno. El LLM se consulta cada diez segundos para decidir la siguiente acción, y el juego se actualiza en consecuencia.

## Aviso
 Lea el link para obtener la clave para groq LLM [enlace](https://www.geeksforgeeks.org/groq-api-with-llama-3/)
## Características

- **Generación de laberintos**: Genera laberintos aleatorios utilizando el algoritmo de retroceso recursivo.
- **Visión por computadora**: Analiza el entorno utilizando técnicas de visión por computadora para detectar paredes y determinar su posición y distancia relativa.
- **Integración con LLM**: Utiliza un LLM para decidir las acciones de navegación basándose en la descripción del entorno.
- **Control de movimiento**: Permite al jugador moverse y girar utilizando comandos específicos.
- **Actualización en tiempo real**: Actualiza la pantalla en tiempo real para reflejar los cambios en la posición y rotación del jugador.

## Requisitos

- Python 3.x
- Pygame
- NumPy
- OpenCV
- Groq (API del LLM)

## Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/tu-usuario/MazeVisionGame.git
   cd MazeVisionGame
   ```

2. **Instalar las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurar la API del LLM**:
   - Obtén una clave de API de Groq desde [Groq Console](https://console.groq.com/docs/overview).
   - Reemplaza `"your_api_key_here"` en el código con tu clave de API.

## Uso

1. **Ejecutar el juego**:
   ```bash
   python maze_vision_game.py
   ```

2. **Controles del juego**:
   - **Moverse hacia adelante**: Tecla `W` o flecha arriba.
   - **Moverse hacia atrás**: Tecla `S` o flecha abajo.
   - **Moverse a la izquierda**: Tecla `A` o flecha izquierda.
   - **Moverse a la derecha**: Tecla `D` o flecha derecha.
   - **Girar a la izquierda**: Tecla `L`.
   - **Girar a la derecha**: Tecla `R`.
   - **Salir del juego**: Tecla `ESC`.

## Funcionamiento

### Inicialización

- **Configuración de Pygame**: Inicializa Pygame y configura la pantalla.
- **Parámetros del juego**: Define los parámetros del juego, como el tamaño del laberinto, la velocidad de movimiento y rotación, y los gradientes del cielo y el suelo.
- **Generación del laberinto**: Genera un laberinto aleatorio utilizando el algoritmo de retroceso recursivo.

### Bucle principal

- **Manejo de eventos**: Maneja los eventos de Pygame, como la salida del juego y los comandos de movimiento.
- **Procesamiento de visión**: Analiza el entorno utilizando técnicas de visión por computadora para detectar paredes y determinar su posición y distancia relativa.
- **Llamada al LLM**: Consulta al LLM cada diez segundos para decidir la siguiente acción basándose en la descripción del entorno.
- **Manejo de movimiento**: Actualiza la posición y rotación del jugador basándose en la acción decidida por el LLM.
- **Renderizado**: Renderiza el frame actual del juego y actualiza la pantalla.

### Funciones principales

- **`reset_game()`**: Reinicia el estado del juego y genera un nuevo laberinto.
- **`generate_maze()`**: Genera un nuevo laberinto aleatorio utilizando el algoritmo de retroceso recursivo.
- **`cast_ray()`**: Lanza un rayo y devuelve la distancia y propiedades de la pared detectada.
- **`handle_movement(action)`**: Maneja el movimiento del jugador y la detección de colisiones basándose en la acción especificada.
- **`render_frame()`**: Renderiza un frame del juego.
- **`process_vision()`**: Procesa el frame actual para el análisis de visión.
- **`print_cv_description()`**: Imprime la descripción generada por la visión por computadora.
- **`call_llm(description)`**: Llama al LLM con la descripción del entorno.
- **`print_llm_response()`**: Imprime la respuesta del LLM.
- **`run()`**: Bucle principal del juego.

## Depuración

- **Impresiones de depuración**: El código incluye impresiones de depuración en `handle_movement` para verificar que los comandos se están ejecutando correctamente y que la posición y rotación del jugador se están actualizando.
- **Verificación de movimiento**: El código asegura que los comandos de movimiento se traduzcan correctamente en cambios en la posición y rotación del jugador.
- **Sensibilidad mejorada**: El código ajusta la detección de paredes para que sea más precisa y sensible al movimiento, filtrando contornos más pequeños.
- **Actualización de la pantalla**: El código asegura que la pantalla se actualice correctamente después de cada iteración del bucle principal.

## Contribución

Las contribuciones son bienvenidas. Por favor, abre un issue o envía un pull request en el repositorio de GitHub.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

## Contacto

Para cualquier pregunta o sugerencia, por favor contacta a [tu-email@example.com](mailto:tu-email@example.com).

[EN]
[EN]

## MazeVisionGame

## Description

MazeVisionGame is a 3D maze game that uses computer vision to analyze the environment and make navigation decisions. The game incorporates a large language model (LLM) to decide what actions to take based on the description of the environment. The LLM is queried every ten seconds to decide the next action, and the game is updated accordingly.

## Alert
Please, read the article for get the API Key from Grow: [link](https://www.geeksforgeeks.org/groq-api-with-llama-3/)
## Features

- Maze generation**: Generates random mazes using the recursive backtracking algorithm.
- Computer Vision**: Analyses the environment using computer vision techniques to detect walls and determine their position and relative distance.
- Integration with LLM**: Uses an LLM to decide navigation actions based on the description of the environment.
- Motion control**: Allows the player to move and turn using specific commands.
- Real-time update**: Updates the screen in real time to reflect changes in the player's position and rotation.

## Requirements

- Python 3.x
- Pygame
- NumPy
- OpenCV
- Groq (LLM API)

## Installation

1. **Clone the repository**:
   ````bash
   git clone https://github.com/tu-usuario/MazeVisionGame.git
   cd MazeVisionGame
   ```

2. **Install the dependencies**:
   ````bash
   pip install -r requirements.txt
   ```

3. **Configure the LLM API**:
   - Get a Groq API key from [Groq Console](https://console.groq.com/docs/overview).
   - Replace `‘your_api_key_here’` in the code with your API key.

## Usage

1. **Run the game**:
   ````bash
   python maze_vision_game.py
   ```

2. **Game controls**:
   - **Move forward**: `W` or up arrow key.
   - Move backwards**: `S` or down arrow key.
   - Move left**: `A` or left arrow key.
   - Move right**: `D` or right arrow key.
   - Turn left**: `L` key.
   - Turn right**: `R` key.
   - Exit the game**: `ESC` key.

## Operation

### Initialisation

- Pygame configuration**: Initialise Pygame and configure the screen.
- Game parameters**: Define game parameters, such as maze size, speed of movement and rotation, and sky and ground gradients.
- Maze generation**: Generates a random maze using the recursive backtracking algorithm.

### Main loop

- Event handling**: Handles Pygame events, such as game exit and movement commands.
- Vision processing**: Analyses the environment using computer vision techniques to detect walls and determine their position and relative distance.
- LLM call**: Queries the LLM every ten seconds to decide the next action based on the description of the environment.
- Motion Management**: Updates player position and rotation based on the action decided by the LLM.
- Rendering**: Renders the current game frame and updates the screen.

### Main functions

- **`reset_game()`**: Resets the game state and generates a new maze.
- **`generate_maze()`**: Generates a new random maze using the recursive backtracking algorithm.
- **`cast_ray()`**: Cast a ray and return the distance and properties of the detected wall.
- **`handle_movement(action)`**: Handles player movement and collision detection based on the specified action.
- **`render_frame()`**: Renders a game frame.
- **`process_vision()`**: Processes the current frame for vision analysis.
- **`print_cv_description()`**: Prints the description generated by the computer vision.
- **`call_llm(description)`**: Calls the LLM with the description of the environment.
- **`print_llm_response()`**: Prints the LLM response.
- **`run()`**: Main game loop.

## Debugging

- **`debug prints**: The code includes debug prints in `handle_movement` to verify that commands are being executed correctly and that player position and rotation are being updated.
- Movement verification**: The code ensures that movement commands are correctly translated into changes in player position and rotation.
- Improved sensitivity**: Code adjusts wall detection to be more accurate and sensitive to motion, filtering out smaller contours.
- **Screen refresh**: The code ensures that the screen refreshes correctly after each iteration of the main loop.

## Contribution

Contributions are welcome. Please open an issue or submit a pull request in the GitHub repository.

## License

This project is licensed under the MIT License. See the [GNU gpl 3](https://www.gnu.org/licenses/gpl-3.0.html) file for details.

## Contact

For any questions or suggestions, please contact [marco.baturan@gmail.com](mailto:tu-email@e