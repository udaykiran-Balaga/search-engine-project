# SubHunt - Subtitle Search Engine

SubHunt is a simple subtitle search engine built with Flask, SQLite, and BERT for natural language processing. It allows users to search for subtitles based on their content and provides relevant search results.

# Features:

- Search for subtitles using a search query
- Display search results with highlighted search terms
- View the full content of a subtitle file

# Installation:

1. Clone the repository:
   git clone https://github.com/your_username/subhunt.git
   cd subhunt

2. Install the required dependencies:
   pip install -r requirements.txt

3. Set up the SQLite database:
   - Create a SQLite database file (chromadb.db in this example) and populate it with subtitle data.
   - Modify the db_file_path variable in app.py to point to your SQLite database file.

4. Run the Flask application:
   python app.py

5. Access the application in your web browser at http://localhost:5000.

# Usage:

1. Enter a search query in the search box on the home page and click the "Search" button.
2. The search results will be displayed, showing the subtitle filenames and the relevant content with highlighted search terms.
3. Click on a subtitle filename to view the full content of the subtitle file.

# License:

This project is licensed under the MIT License - see the LICENSE file for details.
