import pygame as pg
import numpy as np
import cv2
from pygame.locals import *
import time
from groq import Groq

class MazeVisionGame:
    def __init__(self, maze_size=25, screen_width=800, screen_height=600, render_width=120, api_key=None):
        """
        Initialize the 3D maze game with vision analysis.
        """
        # Initialize Pygame
        pg.init()
        pg.mouse.set_visible(True)  # Make the mouse visible

        # Screen setup
        self.screen = pg.display.set_mode((screen_width, screen_height))
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont("Arial", 18)

        # Game parameters
        self.size = maze_size
        self.render_width = render_width
        self.render_height = int(render_width * 0.75)
        self.mod = render_width / 60  # Scaling factor for field of view

        # Movement parameters
        self.move_speed = 0.1
        self.rot_speed = 0.1

        # Initialize game state
        self.reset_game()

        # Precompute gradients for sky and floor
        self._setup_gradients()

        # Timer for the subfunction
        self.last_subfunction_time = time.time()
        self.last_llm_call_time = time.time()
        self.last_cv_description_time = time.time()
        self.last_pos = (self.posx, self.posy)
        self.attempts = 0

        # LLM client
        self.client = self.api_call(api_key)

    def api_call(self, api_key):
        # Instance object Groq
        client = Groq(api_key=api_key)
        return client

    def _setup_gradients(self):
        """Set up sky and floor color gradients."""
        gradient = np.linspace(0, 1, int(self.render_height / 2 - 1))
        self.sky = np.asarray([gradient / 3, gradient / 2 + 0.25, gradient / 3 + 0.5]).T
        self.floor = 0.9 * np.asarray([gradient, gradient, gradient]).T

    def reset_game(self):
        """Reset the game state and generate new maze."""
        self.posx = 1
        self.posy = np.random.randint(1, self.size - 1)
        self.rot = 1
        self.generate_maze()

    def generate_maze(self):
        """Generate a new random maze using recursive backtracking."""
        # Initialize maze with walls
        self.maze_walls = np.ones((self.size, self.size))
        self.maze_heights = np.ones((self.size, self.size))
        self.maze_reflect = np.zeros((self.size, self.size))

        def carve_path(x, y):
            self.maze_walls[x][y] = 0  # Clear current cell

            # Define possible directions (right, down, left, up)
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            np.random.shuffle(directions)

            # Try each direction
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (0 < new_x < self.size - 1 and
                        0 < new_y < self.size - 1 and
                        self.maze_walls[new_x][new_y] == 1):
                    # Carve through the wall between cells
                    self.maze_walls[x + dx // 2][y + dy // 2] = 0
                    carve_path(new_x, new_y)

        # Start carving from a random point
        start_x = 1
        start_y = np.random.randint(1, self.size - 1)
        carve_path(start_x, start_y)

        # Set exit point
        self.exit_x = self.size - 2
        self.exit_y = np.random.randint(1, self.size - 1)
        self.maze_walls[self.exit_x][self.exit_y] = 0

    def cast_ray(self, x, y, angle):
        """Cast a single ray and return wall distance and properties."""
        sin_a = np.sin(angle)
        cos_a = np.cos(angle)

        # Ray step sizes
        dx = np.sqrt(1 + (sin_a / cos_a) ** 2) if cos_a != 0 else 1e30
        dy = np.sqrt(1 + (cos_a / sin_a) ** 2) if sin_a != 0 else 1e30

        # Initial steps
        step_x = 1 if cos_a > 0 else -1
        step_y = 1 if sin_a > 0 else -1

        # Distance to next grid line
        next_x = int(x + 1) - x if cos_a > 0 else x - int(x)
        next_y = int(y + 1) - y if sin_a > 0 else y - int(y)

        # Distance along ray to next x and y grid lines
        len_x = next_x * dx
        len_y = next_y * dy

        while True:
            # Move to next grid line (either x or y)
            if len_x < len_y:
                x += step_x
                if x < 0 or x >= self.size:
                    return float('inf'), 1, [0.3, 0.3, 0.3]
                if self.maze_walls[int(x)][int(y)]:
                    dist = len_x
                    height = self.maze_heights[int(x)][int(y)]
                    color = [0.3, 0.3, 0.3]  # Gray walls
                    return dist, height, color
                len_x += dx
            else:
                y += step_y
                if y < 0 or y >= self.size:
                    return float('inf'), 1, [0.3, 0.3, 0.3]
                if self.maze_walls[int(x)][int(y)]:
                    dist = len_y
                    height = self.maze_heights[int(x)][int(y)]
                    color = [0.4, 0.4, 0.4]  # Slightly lighter for different face
                    return dist, height, color
                len_y += dy

    def handle_movement(self, action):
        """Handle player movement and collision detection."""
        dt = self.clock.tick(60) / 1000.0

        print(f"Action: {action}")  # Debug print

        if action == "go":
            new_x = self.posx + np.cos(self.rot) * self.move_speed * dt
            new_y = self.posy + np.sin(self.rot) * self.move_speed * dt
            if not self.maze_walls[int(new_x)][int(new_y)]:
                self.posx = new_x
                self.posy = new_y
        elif action == "back":
            new_x = self.posx - np.cos(self.rot) * self.move_speed * dt
            new_y = self.posy - np.sin(self.rot) * self.move_speed * dt
            if not self.maze_walls[int(new_x)][int(new_y)]:
                self.posx = new_x
                self.posy = new_y
        elif action == "left":
            new_x = self.posx + np.sin(self.rot) * self.move_speed * dt
            new_y = self.posy - np.cos(self.rot) * self.move_speed * dt
            if not self.maze_walls[int(new_x)][int(new_y)]:
                self.posx = new_x
                self.posy = new_y
        elif action == "right":
            new_x = self.posx - np.sin(self.rot) * self.move_speed * dt
            new_y = self.posy + np.cos(self.rot) * self.move_speed * dt
            if not self.maze_walls[int(new_x)][int(new_y)]:
                self.posx = new_x
                self.posy = new_y
        elif action == "turn left":
            self.rot -= self.rot_speed * dt
        elif action == "turn right":
            self.rot += self.rot_speed * dt

        print(f"New Position: ({self.posx}, {self.posy}), Rotation: {self.rot}")  # Debug print

    def render_frame(self):
        """Render a single frame of the game."""
        pixels = np.zeros([self.render_height, self.render_width, 3])

        # Ray casting loop
        for i in range(self.render_width):
            rot_i = self.rot + np.deg2rad(i / self.mod - 30)

            # Draw sky and floor
            pixels[0:len(self.sky), i] = self.sky * (0.7 + np.sin((rot_i - np.pi / 2) / 2) ** 2 / 3)
            pixels[int(self.render_height / 2):int(self.render_height / 2) + len(self.floor), i] = (
                    self.floor * (0.75 + np.sin((rot_i + np.pi / 2) / 2) ** 2 / 4))

            # Cast ray and draw wall
            dist, height, color = self.cast_ray(self.posx, self.posy, rot_i)

            if dist < float('inf'):
                # Calculate wall height and position
                h = min(1, 1 / (dist + 0.000001))
                wall_start = int((self.render_height - h * self.render_height) / 2)
                wall_end = int((self.render_height + h * self.render_height) / 2)

                # Apply distance shading
                shade = 1 / (1 + dist ** 2 * 0.1)
                wall_color = np.array(color) * shade

                # Draw wall slice
                pixels[wall_start:wall_end, i] = wall_color

        # Scale and display the rendered frame
        surf = pg.surfarray.make_surface(np.rot90(pixels * 255).astype('uint8'))
        surf = pg.transform.scale(surf, (self.screen.get_width(), self.screen.get_height()))
        self.screen.blit(surf, (0, 0))

        # Display FPS
        fps = self.font.render(str(int(self.clock.get_fps())), 1, pg.Color("coral"))
        self.screen.blit(fps, (10, 0))

    def process_vision(self):
        """Process the current frame for vision analysis."""
        # Capture current frame
        frame = pg.surfarray.array3d(self.screen)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Process frame for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects = []

        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Filter small contours, more sensitive
                x, y, w, h = cv2.boundingRect(contour)
                objects.append({
                    'position': (x, y),
                    'size': (w, h),
                    'center': (x + w // 2, y + h // 2)
                })

        # Generate descriptions
        center_x = self.screen.get_width() // 2
        description = []
        for obj in objects:
            x, y = obj['position']
            w, h = obj['size']
            center = obj['center']
            # Determine position relative to view
            if center[0] < center_x - 50:
                position = "left"
            elif center[0] > center_x + 50:
                position = "right"
            else:
                position = "center"
            # Determine approximate distance
            if h > self.screen.get_height() // 3:
                distance = "very close"
            elif h > self.screen.get_height() // 4:
                distance = "close"
            else:
                distance = "far"
            description.append(f"Detected wall in ({x},{y}) at {position} side, {distance}")

        return description

    def print_cv_description(self):
        """Print the description from computer vision."""
        description = self.process_vision()
        print("CV Description:", description)

    def call_llm(self, description):
        """Call the LLM with the description of the environment."""
        prompt = (
            "You are a navigation assistant in a maze. Your task is to decide the action to take based on the description of the environment. "
            "You can only respond with one of the following actions: \"left\", \"right\", \"go\", \"back\", \"turn left\", \"turn right\". "
            "Do not provide any other information, explanation, or additional context. Simply respond with one of the mentioned actions.\n\n"
            "Example input:\n\"Detected wall in (1,10) at left side, far\"\n\n"
            "Example output:\n\"left\"\n\n"
            "Now, based on the following description of the environment, decide the action to take:\n"
        )
        prompt += "\n".join(description)

        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama3-70b-8192"
        )
        return chat_completion.choices[0].message.content.lower()  # Ensure the response is in lowercase

    def print_llm_response(self):
        """Print the response from the LLM."""
        description = self.process_vision()
        response = self.call_llm(description)
        print("LLM Response:", response)
        return response

    def run(self):
        """Main game loop."""
        running = True
        while running:
            # Handle events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False

            # Check if 5 seconds have passed for CV description
            current_time = time.time()
            if current_time - self.last_cv_description_time >= 5:
                self.print_cv_description()
                self.last_cv_description_time = current_time

            # Check if 10 seconds have passed for LLM call
            if current_time - self.last_llm_call_time >= 10:
                action = self.print_llm_response()
                self.handle_movement(action)
                self.last_llm_call_time = current_time

                # Check if there is no movement after three commands
                if (self.posx, self.posy) == self.last_pos:
                    self.attempts += 1
                    if self.attempts >= 3:
                        # Try a different action
                        different_actions = ["left", "right", "go", "back", "turn left", "turn right"]
                        different_actions.remove(action)
                        new_action = different_actions[np.random.randint(0, len(different_actions))]
                        self.handle_movement(new_action)
                        self.attempts = 0  # Reset attempts
                else:
                    self.last_pos = (self.posx, self.posy)
                    self.attempts = 0  # Reset attempts

            # Render frame
            self.render_frame()

            # Check for maze exit
            if int(self.posx) == self.exit_x and int(self.posy) == self.exit_y:
                print("¡Laberinto completado!")
                running = False

            # Update display
            pg.display.flip()

        # Cleanup
        pg.quit()

if __name__ == "__main__":
    api_key = "your_api_key_here"  # Replace with your actual API key
    game = MazeVisionGame(api_key=api_key)
    game.run()
