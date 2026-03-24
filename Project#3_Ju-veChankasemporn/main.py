from VacuumVisualizer import VacuumVisualizer
from VacuumEnvironment import VacuumEnvironment, GRID_ROWS, GRID_COLS

def main():
    env = VacuumEnvironment(rows=GRID_ROWS, cols=GRID_COLS, dirt_probability=0.4)
    visualizer = VacuumVisualizer(env)
    visualizer.run()


if __name__ == "__main__":
    main()