from flask import Flask, request, render_template
from markupsafe import Markup
from transformers import BertTokenizer, BertModel
import torch
import sqlite3
from flask_caching import Cache

app = Flask(__name__)
app.config['CACHE_TYPE'] = 'simple'  # Use a simple in-memory cache
cache = Cache()
cache.init_app(app)

# Load the BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# Path to the SQLite database file
db_file_path = r"D:\experiment\ChromaDB.db"

def get_embeddings(text):
    # Tokenize the text
    tokens = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    # Get the BERT embeddings for the text
    with torch.no_grad():
        outputs = model(**tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze(0)
    return embeddings.numpy()

def cosine_similarity(embedding1, embedding2):
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    magnitude1 = sum(a ** 2 for a in embedding1) ** 0.5
    magnitude2 = sum(a ** 2 for a in embedding2) ** 0.5
    return dot_product / (magnitude1 * magnitude2)

def highlight_query_in_chunk(query, chunk):
    # Replace the query in the chunk with the highlighted version
    highlighted_chunk = chunk.replace(query, f'<mark>{query}</mark>')
    return highlighted_chunk

def get_top_subtitles(query_embeddings, subtitle_embeddings, subtitles, chunks, search_query, k=10):
    similarities = [
        (idx, cosine_similarity(query_embeddings, emb))
        for idx, emb in enumerate(subtitle_embeddings)
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in similarities[:k]]
    # Highlight the query in each chunk and prepend the query to the chunk
    return [(subtitles[idx], Markup(f'<b>{search_query}</b>: {highlight_query_in_chunk(search_query, chunks[idx])}')) for idx in top_indices]

@app.route('/', methods=['GET', 'POST'])
def search():
    search_query = ''
    if request.method == 'POST':
        # Get the search query from the form
        search_query = request.form.get('query')

        # Check if the search results are already cached
        cached_results = cache.get(search_query)
        if cached_results is not None:
            # If cached results are found, return them directly
            return render_template('search.html', subtitles=cached_results, search_query=search_query)

        # Connect to the database
        conn = sqlite3.connect(db_file_path)
        cursor = conn.cursor()

        # Query to select the 'name', 'content', and 'final_embeddings' columns from the 'my_subtitle_table' table
        query = 'SELECT name, content, final_embeddings, chunks FROM my_subtitle_table'

        # Fetch all rows from the table
        cursor.execute(query)
        rows = cursor.fetchall()

        # Close the connection
        conn.close()

        # Extract subtitles, contents, and embeddings from the rows
        subtitles = [row[0] for row in rows]
        embeddings = [eval(row[2]) for row in rows]
        chunks = [row[3] for row in rows]

        # Get the BERT embeddings for the search query
        query_embeddings = get_embeddings(search_query)

        # Get the top k most relevant subtitles
        top_subtitles = get_top_subtitles(query_embeddings, embeddings, subtitles, chunks, search_query)

        # Cache the search results for the entire usage of the application
        cache.set(search_query, top_subtitles)

        # Render the template with the search results
        return render_template('search.html', subtitles=top_subtitles, search_query=search_query)

    # If no search query is provided, render the empty form
    return render_template('search.html', subtitles=[], search_query=search_query)


@app.route('/content')
def content():
    subtitle = request.args.get('subtitle')
    # Fetch the full content of the subtitle file based on the subtitle filename
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    query = 'SELECT content FROM my_subtitle_table WHERE name=?'
    cursor.execute(query, (subtitle,))
    row = cursor.fetchone()
    conn.close()
    full_text_content = row[0] if row else 'Subtitle content not found'

    # Modify the subtitle name to include "Subtitle" before the filename
    heading_subtitle = f"Subtitle {subtitle}"

    return render_template('content.html', subtitle=heading_subtitle, content=full_text_content)

if __name__ == '__main__':
    app.run(debug=True)
