def main():
    print("Hello, World!")

if __name__ == "__main__":
    print(f'this script is being run directly: {__name__}')
    main()
else:
    print("This script is being imported, not run directly:", __name__)