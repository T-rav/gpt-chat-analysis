from configuration import Config
from chat_analysis_options import ChatAnalysisOptions

def main() -> None:
    """Entry point for the chat analysis tool."""
    app = ChatAnalysisOptions()
    app.run()

if __name__ == '__main__':
    main()
