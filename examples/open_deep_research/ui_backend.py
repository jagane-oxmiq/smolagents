from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS
import os
import time
import json
import logging
import datetime
import threading
import textwrap
import argparse
import markdown
from flask import Flask, request, jsonify, send_from_directory, redirect, Response

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
finished_directory = None  # Will be set from command line args
static_directory = None  # Will be set from command line args or default
jobs_directory = None  # Will be set from command line args


def parse_arguments():
    parser = argparse.ArgumentParser(description='Oxmiq DeepInsights Backend Server')
    parser.add_argument('--finished_directory', required=True, help='Directory containing finished research JSON files')
    parser.add_argument('--static_directory', required=False, default='static', help='Directory containing static web files')
    parser.add_argument('--jobs_directory', required=True, help='Directory to write job JSON files')
    return parser.parse_args()


def debug_directory_contents(directory):
    """Print all files in a directory for debugging"""
    logger.info(f"Debugging contents of directory: {directory}")
    
    if not os.path.exists(directory):
        logger.error(f"Directory does not exist: {directory}")
        return
    
    try:
        files = os.listdir(directory)
        logger.info(f"Files in directory ({len(files)} total): {files}")
        
        for filename in files:
            if filename.endswith('.json'):
                file_path = os.path.join(directory, filename)
                try:
                    file_size = os.path.getsize(file_path)
                    logger.info(f"JSON file: {filename}, Size: {file_size} bytes")
                except Exception as e:
                    logger.error(f"Error getting file info for {filename}: {str(e)}")
    except Exception as e:
        logger.error(f"Error listing directory contents: {str(e)}")

def load_completed_questions():
    """Load all completed research questions from the finished directory"""
    completed = []
    
    # Debug directory contents before loading
    debug_directory_contents(finished_directory)
    
    if not os.path.exists(finished_directory):
        logger.warning(f"Finished directory {finished_directory} does not exist")
        return completed
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(finished_directory) if f.endswith('.json')]
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    for filename in json_files:
        file_path = os.path.join(finished_directory, filename)
        try:
            with open(file_path, 'r') as f:
                file_content = f.read()
                logger.debug(f"File content for {filename}: {file_content[:100]}...")
                
                data = json.loads(file_content)
                
                # Check required fields explicitly
                has_question = 'question' in data
                has_answer = 'answer' in data
                has_logs_dir = 'logs_dir' in data
                
                logger.info(f"File {filename} - Has question: {has_question}, Has answer: {has_answer}, Has logs_dir: {has_logs_dir}")
                
                # Ensure the required fields exist
                if has_question and has_answer and has_logs_dir:
                    completed.append(data)
                    logger.info(f"Added completed question from {filename}: {data['question']}")
                else:
                    logger.warning(f"File {filename} missing required fields")
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
    
    logger.info(f"Total completed questions loaded: {len(completed)}")
    return completed


def question_exists(question_text):
    """Check if a question has already been researched"""
    completed = load_completed_questions()
    for item in completed:
        if item['question'].lower() == question_text.lower():
            return item
    return None


def submit_deep_research(question_text):
    """
    Submit a deep research question by writing a JSON file to the jobs directory
    Returns the question_id (timestamp) used for the job file
    """
    # Generate a timestamp to use as question ID
    timestamp = int(time.time())
    question_id = str(timestamp)
    
    # Create a job JSON file
    job_file_name = f"job-{question_id}.json"
    job_file_path = os.path.join(jobs_directory, job_file_name)
    
    # Ensure the jobs directory exists
    os.makedirs(jobs_directory, exist_ok=True)
    
    # Create the job data
    job_data = {
        'question': question_text,
        'timestamp': timestamp,
        'status': 'submitted'
    }
    
    # Write the job file
    try:
        with open(job_file_path, 'w') as f:
            json.dump(job_data, f, indent=2)
        logger.info(f"Job file created: {job_file_path}")
    except Exception as e:
        logger.error(f"Error creating job file: {str(e)}")
        raise
    
    return question_id


def generate_structured_answer(question_text):
    """
    Generate a structured research answer with proper formatting
    This is a simulation - in a real system this would call your research engine
    """
    # Extract keywords from the question to personalize the response
    keywords = extract_keywords(question_text)
    topic = keywords[0] if keywords else "this topic"
    
    # Create a well-formatted answer with multiple paragraphs and structure
    answer = textwrap.dedent(f"""
    Oxmiq DeepInsights analysis: {question_text}

    Our comprehensive analysis reveals several key insights about {topic}. The data indicates significant patterns that merit consideration in any strategic approach to this question.

    Key Findings:
    Based on our multi-source analysis, we've identified three critical factors that influence outcomes in this domain. The primary driver appears to be contextual variables that shift depending on implementation specifics.

    Detailed Analysis:
    The research shows a clear correlation between {keywords[1] if len(keywords) > 1 else "input variables"} and expected outcomes. When examining historical precedents, we observe consistent patterns that support this conclusion.

    Several important considerations emerged during our investigation:
    - The relationship between variables is non-linear in most observed cases
    - External factors can significantly impact results in unpredictable ways
    - Long-term sustainability depends on maintaining system equilibrium

    Strategic Implications:
    Organizations should consider implementing an adaptive approach that accounts for the variability we've observed. The most successful implementations maintain flexibility while establishing clear guiding principles.

    Further research would benefit from deeper investigation into edge cases and exceptional scenarios that might challenge these findings.
    """).strip()
    
    return answer


def extract_keywords(text):
    """Extract potential keywords from the question text"""
    # This is a very simplified keyword extraction - in a real system, 
    # you would use NLP techniques like TF-IDF, NER, etc.
    common_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 
                   'about', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                   'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'why',
                   'how', 'what', 'which', 'who', 'whom', 'whose', 'that', 'this'}
    
    # Split text into words, convert to lowercase
    words = text.lower().replace('?', '').replace('.', '').replace(',', '').split()
    
    # Filter out common words and short words
    keywords = [word for word in words if word not in common_words and len(word) > 3]
    
    # If no keywords found, use some generic terms
    if not keywords:
        keywords = ['topic', 'subject', 'concept', 'scenario']
    
    return keywords

# API Endpoints
@app.route('/api/submit', methods=['POST'])
def submit_question():
    """Submit a new research question"""
    try:
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Check if this question has already been researched
        existing = question_exists(question)
        if existing:
            return jsonify({
                'status': 'cached',
                'question': existing['question'],
                'answer': existing['answer'],
                'logs_dir': existing.get('logs_dir')
            })
        
        # Submit the research job
        question_id = submit_deep_research(question)
        
        return jsonify({
            'status': 'started',
            'question_id': question_id,
            'message': 'Research has been started. Check the status tab for updates.'
        })
        
    except Exception as e:
        logger.error(f"Error submitting question: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/running', methods=['GET'])
def get_running_questions():
    """Get all currently running research questions"""
    try:
        questions_list = []
        
        # Ensure the jobs directory exists
        if not os.path.exists(jobs_directory):
            logger.warning(f"Jobs directory {jobs_directory} does not exist")
            return jsonify({'questions': []})
        
        # Get all JSON files in the jobs directory
        job_files = [f for f in os.listdir(jobs_directory) if f.startswith('job-') and f.endswith('.json')]
        logger.info(f"Found {len(job_files)} job files to process")
        
        for filename in job_files:
            file_path = os.path.join(jobs_directory, filename)
            try:
                with open(file_path, 'r') as f:
                    job_data = json.loads(f.read())
                    
                    # Extract question ID from filename (job-123456789.json -> 123456789)
                    question_id = filename.replace('job-', '').replace('.json', '')
                    
                    # Get the status from job data
                    status = job_data.get('status', 'unknown')
                    
                    # Only include jobs that aren't completed
                    if status != 'completed':
                        # Format the status message based on the status value
                        status_message = "Job submitted - waiting for processing"
                        if status == 'running':
                            status_message = "Research in progress..."
                        elif status == 'failed':
                            status_message = f"Error: {job_data.get('error', 'Research failed')}"
                        
                        # Add job to the questions list
                        ql = {
                            'id': question_id,
                            'question': job_data.get('question', 'Unknown question'),
                            'status': status_message,
                            'started_at': datetime.datetime.fromtimestamp(job_data.get('timestamp', 0)).isoformat()
                        }
                        if 'logs_dir' in job_data:
                            ql['logs_dir'] = job_data['logs_dir']
                        questions_list.append(ql)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in {file_path}: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing job file {file_path}: {str(e)}")
        
        logger.info(f"Returning {len(questions_list)} running/pending questions to client")
        return jsonify({'questions': questions_list})
        
    except Exception as e:
        logger.error(f"Error getting running questions: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/completed', methods=['GET'])
def get_completed_questions():
    """Get all completed research questions"""
    try:
        completed = load_completed_questions()
        logger.info(f"Returning {len(completed)} completed questions to client")
        return jsonify({'questions': completed})
        
    except Exception as e:
        logger.error(f"Error getting completed questions: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/search', methods=['GET'])
def search_questions():
    """Search for research questions by keyword"""
    try:
        query = request.args.get('q', '').lower()
        if not query:
            return jsonify({'error': 'No search query provided'}), 400
        
        # Get all completed questions
        completed = load_completed_questions()
        
        # Filter questions that match the search query
        results = [
            item for item in completed
            if query in item['question'].lower() or query in item['answer'].lower()
        ]
        
        return jsonify({'results': results})
        
    except Exception as e:
        logger.error(f"Error searching questions: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/logs/<path:logs_dir>', methods=['GET'])
def get_logs(logs_dir):
    """Get logs for a specific research question"""
    try:
        # Security check to prevent directory traversal
        if '..' in logs_dir or logs_dir.startswith('/'):
            return jsonify({'error': 'Invalid logs directory'}), 400
        
        # Construct the path to the logs directory
        logs_path = os.path.join(finished_directory, '../research_logs', logs_dir)
        
        if not os.path.exists(logs_path):
            return jsonify({'error': 'Logs not found'}), 404
        
        # Return a simple HTML page with log content
        logs_files = [f for f in os.listdir(logs_path) if f.endswith('.log')]
        
        if not logs_files:
            return "No log files found.", 404
        
        # Read the first log file (in a real implementation, you might want to handle multiple log files)
        log_file_path = os.path.join(logs_path, logs_files[0])
        with open(log_file_path, 'r') as f:
            log_content = f.read()
        
        # Return the log content as plain text
        return log_content, 200, {'Content-Type': 'text/plain'}
        
    except Exception as e:
        logger.error(f"Error getting logs: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/static/logs/<path:path>')
def serve_logs(path):
    print(f"____________________ {path}")
    try:
        # Try to serve the requested file directly
        full_path = os.path.join(static_directory, "logs", path)
        print(f"00000000000000000000 {full_path}")
        if os.path.exists(full_path) and os.path.isfile(full_path):
            print(f"11111111111111111111 {full_path}")
            return send_from_directory(os.path.join(static_directory, 'logs'), path)
        
        # If index.html is requested but doesn't exist, try index.md
        if path == 'index.html' or path.endswith('/index.html'):
            directory = os.path.dirname(path)
            md_path = os.path.join(directory, 'index.md')
            full_md_path = os.path.join(static_directory, md_path)
            
            if os.path.exists(full_md_path) and os.path.isfile(full_md_path):
                logger.info(f"Serving index.md instead of index.html from {md_path}")
                # Read the markdown file
                with open(full_md_path, 'r') as f:
                    md_content = f.read()
                
                # Convert markdown to HTML
                html_content = markdown.markdown(md_content)
                
                # Wrap in a basic HTML template
                final_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oxmiq DeepInsights</title>
    <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>Oxmiq DeepInsights</h1>
            <nav>
                <ul class="tabs">
                    <li><a href="/static/#ask" class="tab-link" data-tab="ask">Ask Question</a></li>
                    <li><a href="/static/#status" class="tab-link" data-tab="status">Research Status</a></li>
                    <li><a href="/static/#completed" class="tab-link" data-tab="completed">Completed Research</a></li>
                </ul>
            </nav>
        </header>

        <main>
            <div class="markdown-content">
                {html_content}
            </div>
        </main>

        <footer>
            <p>&copy; 2025 Oxmiq DeepInsights</p>
        </footer>
    </div>
</body>
</html>"""
                
                return Response(final_html, mimetype='text/html')
        
        # Check if the requested path is a directory and look for index.html or index.md
        if not path.endswith('/'):
            print(f"22222222222222222222 {path}")
            dir_path = os.path.join(static_directory, 'logs', path)
            print(f"55555555555555555555 {dir_path}")
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                # Redirect to directory with trailing slash
                print(f"4444444444444444444 /static/{path}")
                return redirect(f'/static/logs/{path}/')
        else:
            print(f"33333333333333333333 {path}")
            # Path ends with /, check for index.html
            index_html_path = os.path.join(static_directory, 'logs', path, 'index.html')
            if os.path.exists(index_html_path) and os.path.isfile(index_html_path):
                return send_from_directory(os.path.join(static_directory, 'logs', path), 'index.html')
            
            # Check for index.md
            index_md_path = os.path.join(static_directory, 'logs', path, 'index.md')
            if os.path.exists(index_md_path) and os.path.isfile(index_md_path):
                logger.info(f"Serving index.md from directory {path}")
                # Read the markdown file
                with open(index_md_path, 'r') as f:
                    md_content = f.read()
                
                # Convert markdown to HTML
                html_content = markdown.markdown(md_content)
                
                # Wrap in a basic HTML template
                final_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oxmiq DeepInsights</title>
    <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>Oxmiq DeepInsights</h1>
            <nav>
                <ul class="tabs">
                    <li><a href="/static/#ask" class="tab-link" data-tab="ask">Ask Question</a></li>
                    <li><a href="/static/#status" class="tab-link" data-tab="status">Research Status</a></li>
                    <li><a href="/static/#completed" class="tab-link" data-tab="completed">Completed Research</a></li>
                </ul>
            </nav>
        </header>

        <main>
            <div class="markdown-content">
                {html_content}
            </div>
        </main>

        <footer>
            <p>&copy; 2025 Oxmiq DeepInsights</p>
        </footer>
    </div>
</body>
</html>"""
                
                return Response(final_html, mimetype='text/html')
        
        # If we get here, the file was not found
        logger.warning(f"File not found: {path}")
        return f"Error: File not found - {path}", 404
        
    except Exception as e:
        logger.error(f"Error serving static file {path}: {str(e)}")
        return f"Error: {str(e)}", 500


# Static file serving routes
@app.route('/static/', defaults={'path': 'index.html'})
def serve_static(path):
    """Serve static files (HTML, CSS, JS, MD)"""
    try:
        # Try to serve the requested file directly
        full_path = os.path.join(static_directory, path)
        if os.path.exists(full_path) and os.path.isfile(full_path):
            return send_from_directory(static_directory, path)
        
        # If index.html is requested but doesn't exist, try index.md
        if path == 'index.html' or path.endswith('/index.html'):
            directory = os.path.dirname(path)
            md_path = os.path.join(directory, 'index.md')
            full_md_path = os.path.join(static_directory, md_path)
            
            if os.path.exists(full_md_path) and os.path.isfile(full_md_path):
                logger.info(f"Serving index.md instead of index.html from {md_path}")
                # Read the markdown file
                with open(full_md_path, 'r') as f:
                    md_content = f.read()
                
                # Convert markdown to HTML
                html_content = markdown.markdown(md_content)
                
                # Wrap in a basic HTML template
                final_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oxmiq DeepInsights</title>
    <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>Oxmiq DeepInsights</h1>
            <nav>
                <ul class="tabs">
                    <li><a href="/static/#ask" class="tab-link" data-tab="ask">Ask Question</a></li>
                    <li><a href="/static/#status" class="tab-link" data-tab="status">Research Status</a></li>
                    <li><a href="/static/#completed" class="tab-link" data-tab="completed">Completed Research</a></li>
                </ul>
            </nav>
        </header>

        <main>
            <div class="markdown-content">
                {html_content}
            </div>
        </main>

        <footer>
            <p>&copy; 2025 Oxmiq DeepInsights</p>
        </footer>
    </div>
</body>
</html>"""
                
                return Response(final_html, mimetype='text/html')
        
        # Check if the requested path is a directory and look for index.html or index.md
        if not path.endswith('/'):
            dir_path = os.path.join(static_directory, path)
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                # Redirect to directory with trailing slash
                return redirect(f'/static/{path}/')
        else:
            # Path ends with /, check for index.html
            index_html_path = os.path.join(static_directory, path, 'index.html')
            if os.path.exists(index_html_path) and os.path.isfile(index_html_path):
                return send_from_directory(os.path.join(static_directory, path), 'index.html')
            
            # Check for index.md
            index_md_path = os.path.join(static_directory, path, 'index.md')
            if os.path.exists(index_md_path) and os.path.isfile(index_md_path):
                logger.info(f"Serving index.md from directory {path}")
                # Read the markdown file
                with open(index_md_path, 'r') as f:
                    md_content = f.read()
                
                # Convert markdown to HTML
                html_content = markdown.markdown(md_content)
                
                # Wrap in a basic HTML template
                final_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oxmiq DeepInsights</title>
    <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>Oxmiq DeepInsights</h1>
            <nav>
                <ul class="tabs">
                    <li><a href="/static/#ask" class="tab-link" data-tab="ask">Ask Question</a></li>
                    <li><a href="/static/#status" class="tab-link" data-tab="status">Research Status</a></li>
                    <li><a href="/static/#completed" class="tab-link" data-tab="completed">Completed Research</a></li>
                </ul>
            </nav>
        </header>

        <main>
            <div class="markdown-content">
                {html_content}
            </div>
        </main>

        <footer>
            <p>&copy; 2025 Oxmiq DeepInsights</p>
        </footer>
    </div>
</body>
</html>"""
                
                return Response(final_html, mimetype='text/html')
        
        # If we get here, the file was not found
        logger.warning(f"File not found: {path}")
        return f"Error: File not found - {path}", 404
        
    except Exception as e:
        logger.error(f"Error serving static file {path}: {str(e)}")
        return f"Error: {str(e)}", 500


@app.route('/')
def redirect_to_static():
    print(f"++++++++++++++++++++")
    """Redirect root URL to static index page"""
    return redirect('/static/')


# Setup static files if they don't exist
def setup_static_files():
    """Create static files directory and write the frontend files"""
    os.makedirs(static_directory, exist_ok=True)
    
    # Add CSS for markdown content
    css_additions = """
/* Markdown content styling */
.markdown-content {
    padding: 20px;
    background-color: white;
    border-radius: var(--border-radius);
    box-shadow: var(--shadow);
    line-height: 1.6;
}

.markdown-content h1 {
    font-size: 2rem;
    margin-bottom: 20px;
    color: var(--primary-color);
    border-bottom: 2px solid var(--secondary-color);
    padding-bottom: 10px;
}

.markdown-content h2 {
    font-size: 1.5rem;
    margin-top: 25px;
    margin-bottom: 15px;
    color: var(--primary-color);
}

.markdown-content h3 {
    font-size: 1.3rem;
    margin-top: 20px;
    margin-bottom: 10px;
    color: var(--secondary-color);
}

.markdown-content p {
    margin-bottom: 15px;
}

.markdown-content ul, .markdown-content ol {
    margin-left: 20px;
    margin-bottom: 15px;
}

.markdown-content li {
    margin-bottom: 5px;
}

.markdown-content a {
    color: var(--secondary-color);
    text-decoration: none;
}

.markdown-content a:hover {
    text-decoration: underline;
}

.markdown-content code {
    background-color: #f5f5f5;
    padding: 2px 5px;
    border-radius: 3px;
    font-family: monospace;
}

.markdown-content pre {
    background-color: #f5f5f5;
    padding: 15px;
    border-radius: 5px;
    overflow-x: auto;
    margin-bottom: 15px;
}

.markdown-content blockquote {
    border-left: 4px solid var(--secondary-color);
    padding-left: 15px;
    margin-left: 0;
    margin-right: 0;
    font-style: italic;
    color: #555;
}

.markdown-content img {
    max-width: 100%;
    height: auto;
    margin: 15px 0;
    border-radius: var(--border-radius);
}

.markdown-content table {
    border-collapse: collapse;
    width: 100%;
    margin-bottom: 15px;
}

.markdown-content th, .markdown-content td {
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: left;
}

.markdown-content th {
    background-color: #f2f2f2;
    font-weight: bold;
}

.markdown-content tr:nth-child(even) {
    background-color: #f9f9f9;
}
"""
    
    # HTML file
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oxmiq DeepInsights</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>Oxmiq DeepInsights</h1>
            <nav>
                <ul class="tabs">
                    <li><a href="#ask" class="tab-link active" data-tab="ask">Ask Question</a></li>
                    <li><a href="#status" class="tab-link" data-tab="status">Research Status</a></li>
                    <li><a href="#completed" class="tab-link" data-tab="completed">Completed Research</a></li>
                </ul>
            </nav>
        </header>

        <main>
            <!-- Ask Question Tab -->
            <section id="ask" class="tab-content active">
                <h2>Submit a Deep Research Question</h2>
                <form id="question-form">
                    <div class="form-group">
                        <label for="question">Your Research Question:</label>
                        <textarea id="question" name="question" placeholder="Enter your deep research question here..." required></textarea>
                    </div>
                    <button type="submit" class="btn primary-btn">Submit Question</button>
                </form>
                <div id="submission-result" class="result-container"></div>
            </section>

            <!-- Research Status Tab -->
            <section id="status" class="tab-content">
                <h2>Ongoing Research</h2>
                <div class="actions">
                    <button id="refresh-status" class="btn secondary-btn">Refresh Status</button>
                </div>
                <div id="running-questions" class="questions-container">
                    <div class="loading">Loading running research...</div>
                </div>
            </section>

            <!-- Completed Research Tab -->
            <section id="completed" class="tab-content">
                <h2>Completed Research</h2>
                <div class="search-container">
                    <input type="text" id="search-input" placeholder="Search completed research...">
                    <button id="search-btn" class="btn secondary-btn">Search</button>
                </div>
                <div id="completed-questions" class="questions-container">
                    <div class="loading">Loading completed research...</div>
                </div>
            </section>
        </main>

        <footer>
            <p>&copy; 2025 Oxmiq DeepInsights</p>
        </footer>
    </div>

    <script src="script.js"></script>
</body>
</html>"""
    
    # CSS file - updated with better answer formatting and markdown support
    css_content = """/* Base styles */
:root {
    --primary-color: #253746;     /* Darker blue-gray for Oxmiq branding */
    --secondary-color: #1e88e5;   /* Bright blue for Oxmiq branding */
    --accent-color: #e74c3c;
    --light-color: #ecf0f1;
    --dark-color: #2c3e50;
    --success-color: #2ecc71;
    --warning-color: #f39c12;
    --font-main: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    --shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    --border-radius: 5px;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: var(--font-main);
    line-height: 1.6;
    color: var(--dark-color);
    background-color: #f5f7fa;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

h1, h2, h3, h4 {
    color: var(--primary-color);
    margin-bottom: 20px;
}

h1 {
    font-size: 2.2rem;
    border-bottom: 2px solid var(--secondary-color);
    padding-bottom: 10px;
}

h2 {
    font-size: 1.5rem;
}

h3 {
    font-size: 1.2rem;
    margin-bottom: 10px;
}

h4 {
    font-size: 1.1rem;
    margin-bottom: 10px;
    color: var(--secondary-color);
}

a {
    color: var(--secondary-color);
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

/* Navigation and Tabs */
header {
    margin-bottom: 30px;
}

.tabs {
    display: flex;
    list-style: none;
    border-bottom: 1px solid #ddd;
    margin-bottom: 20px;
}

.tab-link {
    display: block;
    padding: 10px 20px;
    color: var(--dark-color);
    text-decoration: none;
    border-bottom: 3px solid transparent;
    transition: all 0.3s ease;
}

.tab-link:hover {
    background-color: rgba(30, 136, 229, 0.1);
    text-decoration: none;
}

.tab-link.active {
    color: var(--secondary-color);
    border-bottom: 3px solid var(--secondary-color);
}

.tab-content {
    display: none;
    padding: 20px;
    background-color: white;
    border-radius: var(--border-radius);
    box-shadow: var(--shadow);
}

.tab-content.active {
    display: block;
}

/* Forms and inputs */
.form-group {
    margin-bottom: 20px;
}

label {
    display: block;
    margin-bottom: 5px;
    font-weight: bold;
}

textarea, input[type="text"] {
    width: 100%;
    padding: 12px;
    border: 1px solid #ddd;
    border-radius: var(--border-radius);
    font-family: var(--font-main);
    font-size: 1rem;
}

textarea {
    height: 150px;
    resize: vertical;
}

.btn {
    display: inline-block;
    padding: 10px 20px;
    background-color: var(--secondary-color);
    color: white;
    border: none;
    border-radius: var(--border-radius);
    cursor: pointer;
    font-size: 1rem;
    transition: background-color 0.3s ease;
}

.btn:hover {
    opacity: 0.9;
}

.primary-btn {
    background-color: var(--secondary-color);
}

.secondary-btn {
    background-color: var(--dark-color);
}

/* Question cards and containers */
.questions-container {
    margin-top: 20px;
}

.question-card {
    background-color: white;
    border-radius: var(--border-radius);
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: var(--shadow);
    border-left: 4px solid var(--secondary-color);
}

.question-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
}

.question-title {
    flex: 1;
    margin-bottom: 0;
}

.question-status {
    display: inline-block;
    padding: 5px 15px;
    border-radius: 15px;
    font-size: 0.8rem;
    font-weight: bold;
    text-transform: uppercase;
}

.status-pending {
    background-color: var(--warning-color);
    color: white;
}

.status-complete {
    background-color: var(--success-color);
    color: white;
}

.question-body {
    margin-top: 15px;
}

/* Improved Answer Container Styling */
.question-answer {
    background-color: var(--light-color);
    padding: 15px;
    border-radius: var(--border-radius);
    margin-top: 15px;
    line-height: 1.6;
}

.question-answer h4 {
    margin-bottom: 12px;
    font-weight: 600;
    color: var(--secondary-color);
    border-bottom: 1px solid rgba(0,0,0,0.1);
    padding-bottom: 5px;
}

.question-answer p {
    margin-bottom: 15px;
    white-space: pre-line; /* Preserve line breaks */
}

.question-answer p:last-child {
    margin-bottom: 0;
}

.question-answer ul, .question-answer ol {
    margin-left: 20px;
    margin-bottom: 15px;
}

.question-answer li {
    margin-bottom: 5px;
}

.question-answer .insight-title {
    font-weight: 600;
    margin-top: 15px;
    color: var(--primary-color);
}

.question-answer .highlight {
    background-color: rgba(30, 136, 229, 0.1);
    padding: 2px 4px;
    border-radius: 3px;
}

.question-answer .key-finding {
    border-left: 3px solid var(--secondary-color);
    padding-left: 10px;
    margin: 15px 0;
    font-style: italic;
}

.question-answer .reference {
    color: #666;
    font-size: 0.9em;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px dashed #ddd;
}

.question-logs {
    margin-top: 15px;
}

.log-link {
    display: inline-flex;
    align-items: center;
    color: var(--secondary-color);
}

.log-link:before {
    content: "📋";
    margin-right: 5px;
}

/* Search container */
.search-container {
    display: flex;
    margin-bottom: 20px;
}

.search-container input {
    flex: 1;
    margin-right: 10px;
}

/* Message styling */
.message {
    padding: 15px;
    border-radius: var(--border-radius);
    margin-bottom: 20px;
}

.message-info {
    background-color: rgba(30, 136, 229, 0.1);
    border-left: 4px solid var(--secondary-color);
}

.message-success {
    background-color: rgba(46, 204, 113, 0.1);
    border-left: 4px solid var(--success-color);
}

.message-warning {
    background-color: rgba(243, 156, 18, 0.1);
    border-left: 4px solid var(--warning-color);
}

.message-error {
    background-color: rgba(231, 76, 60, 0.1);
    border-left: 4px solid var(--accent-color);
}

/* Loading indicator */
.loading {
    text-align: center;
    padding: 20px;
    color: #777;
    font-style: italic;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 40px 20px;
    color: #777;
    border: 2px dashed #ddd;
    border-radius: var(--border-radius);
    margin-top: 20px;
}

.empty-state p {
    margin-bottom: 10px;
}

/* Footer */
footer {
    margin-top: 40px;
    text-align: center;
    color: #777;
    font-size: 0.9rem;
}

/* Markdown content styling */
.markdown-content {
    padding: 20px;
    background-color: white;
    border-radius: var(--border-radius);
    box-shadow: var(--shadow);
    line-height: 1.6;
}

.markdown-content h1 {
    font-size: 2rem;
    margin-bottom: 20px;
    color: var(--primary-color);
    border-bottom: 2px solid var(--secondary-color);
    padding-bottom: 10px;
}

.markdown-content h2 {
    font-size: 1.5rem;
    margin-top: 25px;
    margin-bottom: 15px;
    color: var(--primary-color);
}

.markdown-content h3 {
    font-size: 1.3rem;
    margin-top: 20px;
    margin-bottom: 10px;
    color: var(--secondary-color);
}

.markdown-content p {
    margin-bottom: 15px;
}

.markdown-content ul, .markdown-content ol {
    margin-left: 20px;
    margin-bottom: 15px;
}

.markdown-content li {
    margin-bottom: 5px;
}

.markdown-content a {
    color: var(--secondary-color);
    text-decoration: none;
}

.markdown-content a:hover {
    text-decoration: underline;
}

.markdown-content code {
    background-color: #f5f5f5;
    padding: 2px 5px;
    border-radius: 3px;
    font-family: monospace;
}

.markdown-content pre {
    background-color: #f5f5f5;
    padding: 15px;
    border-radius: 5px;
    overflow-x: auto;
    margin-bottom: 15px;
}

.markdown-content blockquote {
    border-left: 4px solid var(--secondary-color);
    padding-left: 15px;
    margin-left: 0;
    margin-right: 0;
    font-style: italic;
    color: #555;
}

.markdown-content img {
    max-width: 100%;
    height: auto;
    margin: 15px 0;
    border-radius: var(--border-radius);
}

.markdown-content table {
    border-collapse: collapse;
    width: 100%;
    margin-bottom: 15px;
}

.markdown-content th, .markdown-content td {
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: left;
}

.markdown-content th {
    background-color: #f2f2f2;
    font-weight: bold;
}

.markdown-content tr:nth-child(even) {
    background-color: #f9f9f9;
}

/* Responsive styles */
@media (max-width: 768px) {
    .tabs {
        flex-direction: column;
    }
    
    .tabs li {
        margin-bottom: 5px;
    }
    
    .tab-link {
        border-left: 3px solid transparent;
        border-bottom: none;
    }
    
    .tab-link.active {
        border-left: 3px solid var(--secondary-color);
        border-bottom: none;
    }
    
    .search-container {
        flex-direction: column;
    }
    
    .search-container input {
        margin-right: 0;
        margin-bottom: 10px;
    }
    
    .question-header {
        flex-direction: column;
        align-items: flex-start;
    }
    
    .question-status {
        margin-top: 10px;
    }
}"""
    
    # JavaScript file - updated with better answer formatting
    js_content = """// Configuration
const API_BASE_URL = '/api';
const APP_NAME = 'Oxmiq DeepInsights';

// DOM Elements
const elements = {
    // Tabs
    tabLinks: document.querySelectorAll('.tab-link'),
    tabContents: document.querySelectorAll('.tab-content'),
    
    // Forms and inputs
    questionForm: document.getElementById('question-form'),
    questionInput: document.getElementById('question'),
    searchInput: document.getElementById('search-input'),
    searchBtn: document.getElementById('search-btn'),
    
    // Result containers
    submissionResult: document.getElementById('submission-result'),
    runningQuestions: document.getElementById('running-questions'),
    completedQuestions: document.getElementById('completed-questions'),
    
    // Buttons
    refreshStatusBtn: document.getElementById('refresh-status')
};

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Set page title
    document.title = APP_NAME;
    
    // Tab switching
    elements.tabLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            switchTab(link.dataset.tab);
        });
    });
    
    // Form submission
    elements.questionForm.addEventListener('submit', (e) => {
        e.preventDefault();
        submitQuestion();
    });
    
    // Search functionality
    elements.searchBtn.addEventListener('click', () => {
        searchCompletedQuestions();
    });
    
    // Enter key for search
    elements.searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            searchCompletedQuestions();
        }
    });
    
    // Refresh status
    elements.refreshStatusBtn.addEventListener('click', () => {
        loadRunningQuestions();
    });
    
    // Check URL hash for tab navigation
    checkUrlHash();
    
    // Load initial data for active tab
    loadDataForCurrentTab();
});

// Handle browser navigation
window.addEventListener('hashchange', checkUrlHash);

// Functions
function checkUrlHash() {
    const hash = window.location.hash.substring(1);
    if (hash && ['ask', 'status', 'completed'].includes(hash)) {
        switchTab(hash);
    }
}

function switchTab(tabId) {
    // Update URL hash
    window.location.hash = `#${tabId}`;
    
    // Update active tab
    elements.tabLinks.forEach(link => {
        if (link.dataset.tab === tabId) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
    
    // Show active content
    elements.tabContents.forEach(content => {
        if (content.id === tabId) {
            content.classList.add('active');
        } else {
            content.classList.remove('active');
        }
    });
    
    // Load data for the active tab
    loadDataForCurrentTab();
}

function loadDataForCurrentTab() {
    const activeTab = document.querySelector('.tab-content.active').id;
    
    switch (activeTab) {
        case 'status':
            loadRunningQuestions();
            break;
        case 'completed':
            loadCompletedQuestions();
            break;
        // No data loading needed for 'ask' tab
    }
}

async function submitQuestion() {
    const question = elements.questionInput.value.trim();
    
    if (!question) {
        showMessage(elements.submissionResult, 'Please enter a research question', 'error');
        return;
    }
    
    showMessage(elements.submissionResult, 'Submitting question...', 'info');
    
    try {
        const response = await fetch(`${API_BASE_URL}/submit`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            if (data.status === 'cached') {
                // Question already researched
                const resultHtml = `
                    <div class="question-card">
                        <div class="question-header">
                            <h3 class="question-title">${escapeHtml(data.question)}</h3>
                            <span class="question-status status-complete">Already Researched</span>
                        </div>
                        <div class="question-answer">
                            <h4>Insights</h4>
                            ${formatAnswer(data.answer)}
                        </div>
                        ${data.logs_dir ? `
                        <div class="question-logs">
                            <a href="logs/${encodeURIComponent(data.logs_dir)}" class="log-link" target="_blank">View Research Logs</a>
                        </div>
                        ` : ''}
                    </div>
                `;
                elements.submissionResult.innerHTML = resultHtml;
                
            } else {
                // Research started
                showMessage(
                    elements.submissionResult, 
                    `${APP_NAME} has begun researching: "${escapeHtml(question)}"<br>
                    Check the <a href="#status" class="tab-switch">Status tab</a> for updates.`, 
                    'success'
                );
                
                // Clear the form
                elements.questionInput.value = '';
                
                // Add click event to the status tab link
                const tabSwitchLink = elements.submissionResult.querySelector('.tab-switch');
                if (tabSwitchLink) {
                    tabSwitchLink.addEventListener('click', (e) => {
                        e.preventDefault();
                        switchTab('status');
                    });
                }
            }
        } else {
            showMessage(elements.submissionResult, `Error: ${data.error || 'Failed to submit question'}`, 'error');
        }
    } catch (error) {
        showMessage(elements.submissionResult, `Error: ${error.message || 'Failed to connect to server'}`, 'error');
    }
}

async function loadRunningQuestions() {
    elements.runningQuestions.innerHTML = '<div class="loading">Loading active research...</div>';
    
    try {
        const response = await fetch(`${API_BASE_URL}/running`);
        const data = await response.json();
        
        if (response.ok) {
            if (data.questions && data.questions.length > 0) {
                const questionsHtml = data.questions.map(item => `
                    <div class="question-card">
                        <div class="question-header">
                            <h3 class="question-title">${escapeHtml(item.question)}</h3>
                            <span class="question-status status-pending">In Progress</span>
                        </div>
                        <div class="question-body">
                            <p><strong>Status:</strong> ${escapeHtml(item.status)}</p>
                            <p><strong>Started:</strong> ${formatDate(item.started_at)}</p>
                        </div>
                        ${item.logs_dir ? `
                        <div class="question-logs">
                            <a href="logs/${encodeURIComponent(item.logs_dir)}" class="log-link" target="_blank">View Research Logs</a>
                        </div>
                        ` : ''}
                    </div>
                `).join('');
                
                elements.runningQuestions.innerHTML = questionsHtml;
            } else {
                elements.runningQuestions.innerHTML = `
                    <div class="empty-state">
                        <p>No research questions currently in progress</p>
                        <p>Submit a question in the Ask Question tab to get started</p>
                    </div>
                `;
            }
        } else {
            showMessage(elements.runningQuestions, `Error: ${data.error || 'Failed to load running questions'}`, 'error');
        }
    } catch (error) {
        showMessage(elements.runningQuestions, `Error: ${error.message || 'Failed to connect to server'}`, 'error');
    }
}

async function loadCompletedQuestions() {
    elements.completedQuestions.innerHTML = '<div class="loading">Loading completed insights...</div>';
    
    try {
        const response = await fetch(`${API_BASE_URL}/completed`);
        const data = await response.json();
        
        if (response.ok) {
            if (data.questions && data.questions.length > 0) {
                const questionsHtml = data.questions.map(item => createQuestionCard(item)).join('');
                elements.completedQuestions.innerHTML = questionsHtml;
            } else {
                elements.completedQuestions.innerHTML = `
                    <div class="empty-state">
                        <p>No completed research questions found</p>
                        <p>Go to the Ask Question tab to submit a new query</p>
                    </div>
                `;
            }
        } else {
            showMessage(elements.completedQuestions, `Error: ${data.error || 'Failed to load completed questions'}`, 'error');
        }
    } catch (error) {
        showMessage(elements.completedQuestions, `Error: ${error.message || 'Failed to connect to server'}`, 'error');
    }
}

async function searchCompletedQuestions() {
    const searchTerm = elements.searchInput.value.trim();
    
    if (!searchTerm) {
        showMessage(elements.completedQuestions, 'Please enter a search term', 'warning');
        return;
    }
    
    elements.completedQuestions.innerHTML = '<div class="loading">Searching completed insights...</div>';
    
    try {
        const response = await fetch(`${API_BASE_URL}/search?q=${encodeURIComponent(searchTerm)}`);
        const data = await response.json();
        
        if (response.ok) {
            if (data.results && data.results.length > 0) {
                const resultsHtml = `
                    <div class="message message-info">
                        Found ${data.results.length} result${data.results.length > 1 ? 's' : ''} for "${escapeHtml(searchTerm)}"
                    </div>
                    ${data.results.map(item => createQuestionCard(item)).join('')}
                `;
                elements.completedQuestions.innerHTML = resultsHtml;
            } else {
                elements.completedQuestions.innerHTML = `
                    <div class="message message-warning">
                        No results found for "${escapeHtml(searchTerm)}"
                    </div>
                    <button class="btn secondary-btn" id="reset-search">Show All Completed Research</button>
                `;
                
                // Add event listener to reset search button
                document.getElementById('reset-search').addEventListener('click', () => {
                    elements.searchInput.value = '';
                    loadCompletedQuestions();
                });
            }
        } else {
            showMessage(elements.completedQuestions, `Error: ${data.error || 'Failed to search questions'}`, 'error');
        }
    } catch (error) {
        showMessage(elements.completedQuestions, `Error: ${error.message || 'Failed to connect to server'}`, 'error');
    }
}

// Helper Functions
function createQuestionCard(item) {
    return `
        <div class="question-card">
            <div class="question-header">
                <h3 class="question-title">${escapeHtml(item.question)}</h3>
                <span class="question-status status-complete">Completed</span>
            </div>
            <div class="question-answer">
                <h4>Insights</h4>
                ${formatAnswer(item.answer)}
            </div>
            ${item.logs_dir ? `
            <div class="question-logs">
                <a href="logs/${encodeURIComponent(item.logs_dir)}" class="log-link" target="_blank">View Research Logs</a>
            </div>
            ` : ''}
        </div>
    `;
}

// Format answer text to improve readability
function formatAnswer(text) {
    if (!text) return '<p>No answer provided.</p>';
    
    // Process text to identify parts and structure
    let formattedText = text;
    
    // Detect if the text contains a header pattern like "Oxmiq DeepInsights analysis:"
    if (formattedText.includes('Oxmiq DeepInsights analysis:')) {
        // Extract the question part
        const parts = formattedText.split('Oxmiq DeepInsights analysis:');
        if (parts.length > 1) {
            // Remove the question part and any extra whitespace
            formattedText = parts[1].trim();
        }
    }
    
    // Split into paragraphs and format each
    let paragraphs = formattedText.split(/\\n{2,}/g); // Split on double newlines or more
    
    // Process each paragraph
    paragraphs = paragraphs.map(paragraph => {
        // Trim whitespace
        paragraph = paragraph.trim();
        
        // Skip empty paragraphs
        if (!paragraph) return '';
        
        // If paragraph starts with a bullet point marker, format as a list item
        if (paragraph.startsWith('- ') || paragraph.startsWith('• ')) {
            return `<li>${paragraph.substring(2)}</li>`;
        }
        
        // Check if this is a "Key finding" or similar highlighted section
        if (paragraph.toLowerCase().includes('key finding') || 
            paragraph.toLowerCase().includes('important:') ||
            paragraph.toLowerCase().includes('note:')) {
            return `<div class="key-finding">${paragraph}</div>`;
        }
        
        // Check if this is a section header (shorter text ending with a colon)
        if (paragraph.length < 50 && paragraph.endsWith(':')) {
            return `<div class="insight-title">${paragraph}</div>`;
        }
        
        // Regular paragraph
        return `<p>${paragraph}</p>`;
    });
    
    // Join all processed paragraphs
    let result = paragraphs.join('');
    
    // Wrap in a list if multiple list items were detected
    if (result.includes('<li>')) {
        const listItems = result.split('<li>').filter(item => item.includes('</li>'));
        if (listItems.length > 1) {
            // Extract the list items
            const listContent = result.match(/<li>.*?<\\/li>/g).join('');
            // Remove the list items from the original content
            result = result.replace(/<li>.*?<\\/li>/g, '');
            // Add the list items in a proper list
            result = result + `<ul>${listContent}</ul>`;
        }
    }
    
    // If result is empty (after all processing), return a default message
    if (!result.trim()) {
        return '<p>Analysis complete. See logs for details.</p>';
    }
    
    return result;
}

function showMessage(container, message, type = 'info') {
    container.innerHTML = `<div class="message message-${type}">${message}</div>`;
    
    // If there are any links in the message, add event listeners
    const links = container.querySelectorAll('a');
    links.forEach(link => {
        if (link.classList.contains('tab-switch')) {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const tabId = link.getAttribute('href').substring(1);
                switchTab(tabId);
            });
        }
    });
}

function formatDate(dateString) {
    try {
        const date = new Date(dateString);
        return date.toLocaleString();
    } catch (e) {
        return dateString;
    }
}

function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}"""
    
    # Write files to static directory
    with open(os.path.join(static_directory, 'index.html'), 'w') as f:
        f.write(html_content)
        
    with open(os.path.join(static_directory, 'styles.css'), 'w') as f:
        f.write(css_content)
        
    with open(os.path.join(static_directory, 'script.js'), 'w') as f:
        f.write(js_content)
    
    # Create a sample Markdown file for testing
    sample_md_content = """# Welcome to Oxmiq DeepInsights

## About Our Platform

Oxmiq DeepInsights is an advanced research platform that leverages cutting-edge AI technology to provide comprehensive answers to complex research questions.

### Key Features

- **Deep Research**: Our system delves into multiple sources to gather comprehensive information
- **Pattern Analysis**: Advanced algorithms identify patterns and relationships in the data
- **Insight Synthesis**: AI-powered systems combine information to generate actionable insights

## How to Use

1. Navigate to the "Ask Question" tab
2. Enter your research question in the text area
3. Submit your question and wait for our system to process it
4. View your results in the "Completed Research" tab

## Sample Research Topics

Our system can handle a wide variety of research questions across different domains:

```
What factors influence renewable energy adoption in developing economies?
How is artificial intelligence changing healthcare delivery systems?
What are the most effective strategies for urban planning in coastal cities?
```

Feel free to explore our platform and discover the power of AI-driven research!
"""
    
    # Create a docs directory with markdown file
    docs_dir = os.path.join(static_directory, 'docs')
    os.makedirs(docs_dir, exist_ok=True)
    
    with open(os.path.join(docs_dir, 'index.md'), 'w') as f:
        f.write("""# Documentation

## Getting Started

Welcome to the Oxmiq DeepInsights documentation. This guide will help you get familiar with our platform.

### Installation

Our platform is web-based, so there's no installation required. Simply navigate to the main page to begin.

### Using the API

For advanced users, we provide a REST API for programmatic access to our research capabilities:

```python
import requests

def submit_research(question):
    response = requests.post(
        "https://api.oxmiq.com/submit",
        json={"question": question}
    )
    return response.json()

# Example usage
result = submit_research("What are the impacts of climate change on global food security?")
print(result)
```

## Advanced Features

Explore our advanced features in the sections below.
""")
    
    logger.info(f"Static files created in {static_directory}")
    logger.info(f"Sample Markdown files created for testing")


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()
    finished_directory = args.finished_directory
    static_directory = args.static_directory
    jobs_directory = args.jobs_directory
    
    logger.info(f"Starting Oxmiq DeepInsights server with finished directory: {finished_directory}")
    logger.info(f"Static files will be served from: {static_directory}")
    logger.info(f"Job files will be written to: {jobs_directory}")
    
    # Ensure the specified directories exist
    if not os.path.exists(finished_directory):
        logger.warning(f"Creating finished directory: {finished_directory}")
        os.makedirs(finished_directory, exist_ok=True)
    
    if not os.path.exists(jobs_directory):
        logger.warning(f"Creating jobs directory: {jobs_directory}")
        os.makedirs(jobs_directory, exist_ok=True)
    
    # Add markdown and pkg installation check
    try:
        import markdown
    except ImportError:
        logger.warning("Markdown package not found, installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'markdown'])
        import markdown
        logger.info("Markdown package installed successfully")
    
    # Setup static files
    setup_static_files()
    
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
