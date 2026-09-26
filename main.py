import config


def main() -> None:
    print("DriveSense config loaded.")
    print(f"TIGER_DATA_DSN set: {config.TIGER_DATA_DSN is not None}")
    print(f"ELEVENLABS_API_KEY set: {config.ELEVENLABS_API_KEY is not None}")


if __name__ == "__main__":
    main()
