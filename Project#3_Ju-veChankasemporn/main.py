from VacuumVisualizer import VacuumVisualizer
from VacuumEnvironment import VacuumEnvironment


def main():
    env = VacuumEnvironment(rows=5, cols=5, dirt_probability=0.4)
    visualizer = VacuumVisualizer(env)
    visualizer.run()


if __name__ == "__main__":
    main()