# dashboard_server.py - Simple web server to display training progress
import http.server
import socketserver
import os
import json
import time
import threading
from datetime import datetime
import torch

# Configuration
PORT = 8000
MONITOR_DIR = "monitor_plots"
CHECKPOINTS = {
    "classifier": "checkpoints/classifier_checkpoint.pt",
    "denoiser": "checkpoints/denoiser_checkpoint.pt"
}
UPDATE_INTERVAL = 60  # seconds

# Create monitor directory if it doesn't exist
os.makedirs(MONITOR_DIR, exist_ok=True)

# HTML template for the dashboard
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Project 4 Training Progress</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1, h2 {
            color: #333;
        }
        .status {
            padding: 10px;
            background-color: #e8f5e9;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .metrics {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 0 5px rgba(0,0,0,0.1);
            min-width: 200px;
        }
        .plots img {
            max-width: 100%;
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .refresh {
            color: #666;
            font-size: 0.8em;
            margin-top: 40px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Project 4 Training Progress</h1>
        
        <div class="status">
            <h2>Training Status</h2>
            <p>Last updated: {last_updated}</p>
            {status_html}
        </div>
        
        <div class="metrics">
            {metrics_html}
        </div>
        
        <h2>Training Curves</h2>
        <div class="plots">
            <img src="latest_training_curves.png?t={timestamp}" alt="Training Curves">
        </div>
        
        <h2>Latest Generated Samples</h2>
        <div class="plots">
            <img src="latest_samples.png?t={timestamp}" alt="Generated Samples">
        </div>
        
        <p class="refresh">This page auto-refreshes every 60 seconds. Last refreshed: {last_updated}</p>
        
        <script>
            setTimeout(function() {{
                location.reload();
            }}, 60000);
        </script>
    </div>
</body>
</html>
"""

def load_checkpoint_data():
    """Load data from checkpoints and return formatted status info"""
    status = {}
    metrics = {}
    
    for name, path in CHECKPOINTS.items():
        if not os.path.exists(path):
            status[name] = f"Not started yet (checkpoint not found)"
            continue
            
        try:
            checkpoint = torch.load(path, map_location='cpu')
            epoch = checkpoint.get('epoch', 0)
            total_epochs = checkpoint.get('total_epochs', 0) or 200  # Default to 200 if not found
            
            # Calculate progress
            progress_pct = round((epoch / total_epochs) * 100, 1)
            status[name] = f"Epoch {epoch}/{total_epochs} ({progress_pct}% complete)"
            
            # Extract latest metrics
            if 'train_loss_list' in checkpoint and len(checkpoint['train_loss_list']) > 0:
                metrics[f"{name}_loss"] = checkpoint['train_loss_list'][-1]
                
            if 'nll_list' in checkpoint and len(checkpoint['nll_list']) > 0:
                metrics[f"{name}_nll"] = checkpoint['nll_list'][-1]
        except Exception as e:
            status[name] = f"Error loading checkpoint: {e}"
    
    # Format HTML for status
    status_html = "<ul>"
    for name, status_text in status.items():
        status_html += f"<li><strong>{name}:</strong> {status_text}</li>"
    status_html += "</ul>"
    
    # Format HTML for metrics
    metrics_html = ""
    for name, value in metrics.items():
        metrics_html += f"""
        <div class="metric-card">
            <h3>{name.replace('_', ' ').title()}</h3>
            <p>{value:.6f}</p>
        </div>
        """
    
    return status_html, metrics_html

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=MONITOR_DIR, **kwargs)
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Generate dynamic content
            last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            status_html, metrics_html = load_checkpoint_data()
            
            # Fill template
            html = HTML_TEMPLATE.format(
                last_updated=last_updated,
                status_html=status_html,
                metrics_html=metrics_html,
                timestamp=int(time.time())
            )
            
            self.wfile.write(html.encode())
        else:
            super().do_GET()

def run_server():
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"Serving dashboard at http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    print(f"Starting dashboard server on port {PORT}")
    print(f"Open http://localhost:{PORT} in your browser to view training progress")
    
    # Start server in a thread
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Dashboard server stopped by user.") 