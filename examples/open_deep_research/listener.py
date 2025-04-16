#!/usr/bin/env python3
import os
import json
import time
from datetime import datetime
import pyinotify
import argparse
from pathlib import Path
from run_ox import create_agent


class DirectoryMonitor:
    def __init__(self, directory_path, destination_path,
                    model_id, chromadb_host, chromadb_port, chromadb_collection,
                    local_dir, llm_host, llm_port, logs):
        self.directory_path = Path(directory_path)
        self.destination_path = Path(destination_path)
        self.model_id = model_id
        self.chromadb_host = chromadb_host
        self.chromadb_port = chromadb_port
        self.chromadb_collection = chromadb_collection
        self.local_dir = local_dir
        self.llm_host = llm_host
        self.llm_port = llm_port
        self.logs = logs
        
        # Validate source directory
        if not self.directory_path.exists():
            raise FileNotFoundError(f"Directory {directory_path} does not exist")
        if not self.directory_path.is_dir():
            raise NotADirectoryError(f"{directory_path} is not a directory")
            
        # Validate destination directory
        if not self.destination_path.exists():
            print(f"Creating destination directory: {destination_path}")
            self.destination_path.mkdir(parents=True)
        elif not self.destination_path.is_dir():
            raise NotADirectoryError(f"{destination_path} is not a directory")
        
        # Initialize pyinotify
        self.wm = pyinotify.WatchManager()
        self.mask = pyinotify.IN_MODIFY | pyinotify.IN_CREATE | pyinotify.IN_MOVED_TO
        
        # Setup event handler
        self.handler = DirectoryEventHandler(self)
        self.notifier = pyinotify.Notifier(self.wm, self.handler)
        
        # Start watching the directory
        self.wdd = self.wm.add_watch(str(self.directory_path), self.mask, rec=False)
        
        print(f"Monitoring directory: {self.directory_path}")
        print(f"Processed files will be moved to: {self.destination_path}")
    
    def find_oldest_json_file(self):
        """Find the oldest JSON file in the monitored directory."""
        json_files = []
        
        for file_path in self.directory_path.glob("*.json"):
            if file_path.is_file():
                json_files.append((file_path, file_path.stat().st_mtime))
        
        if not json_files:
            print("No JSON files found in the directory.")
            return None
        
        # Sort by modification time (oldest first)
        json_files.sort(key=lambda x: x[1])
        oldest_file = json_files[0][0]
        
        print(f"Oldest JSON file: {oldest_file} (modified: {datetime.fromtimestamp(json_files[0][1])})")
        return oldest_file
    
    def process_json_file(self, file_path):
        """Process the JSON file and move it to the destination directory when complete."""
        print(f"Processing file: {file_path}")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            print(f"File contains {len(data)} items" if isinstance(data, list) else "Processing JSON object")
            print(f"JSON structure: {json.dumps(data, indent=2)[:200]}...")
            
            full_logs_dir = os.path.join(self.logs, f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
            os.makedirs(full_logs_dir)
            agent = create_agent(self.chromadb_host, self.chromadb_port, self.chromadb_collection, self.local_dir,
                                    self.llm_host, int(self.llm_port), full_logs_dir, model_id=self.model_id)

            answer = agent.run(data['question'])

            print(f"Got this answer: {answer}")
            data['answer'] = answer
            data['logs_dir'] = full_logs_dir
            
            print(f"Successfully processed {file_path}")
            
            # Move the file to the destination directory
            dest_file_path = self.destination_path / Path(file_path).name
            
            # Check if file with same name already exists in destination
            if dest_file_path.exists():
                # Append timestamp to make the filename unique
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = Path(file_path).stem + f"_{timestamp}" + Path(file_path).suffix
                dest_file_path = self.destination_path / filename
            
            with open(dest_file_path, 'w') as wfp:
                wfp.write(json.dumps(data, indent=2))

            # Move the file
            os.remove(file_path)
            print(f"Removed processed file. Wrote results to: {dest_file_path}")
            
            return True
        
        except json.JSONDecodeError:
            print(f"Error: {file_path} is not a valid JSON file")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
        
        return False
    
    def start_monitoring(self):
        """Start the monitoring loop."""
        try:
            print("Starting monitor. Press Ctrl+C to stop.")
            # Check once at startup
            self.list_directory()
            self.process_oldest_json()
            
            # Then continue monitoring for changes
            self.notifier.loop()
        except KeyboardInterrupt:
            print("Monitoring stopped.")
        finally:
            # Clean up
            self.notifier.stop()
    
    def list_directory(self):
        """List all files in the directory."""
        print("\nDirectory listing:")
        for item in sorted(self.directory_path.iterdir()):
            mod_time = datetime.fromtimestamp(item.stat().st_mtime)
            file_type = "DIR" if item.is_dir() else "FILE"
            print(f"{file_type} | {mod_time} | {item.name}")
    
    def process_oldest_json(self):
        """Find and process the oldest JSON file."""
        oldest_file = self.find_oldest_json_file()
        if oldest_file:
            self.process_json_file(oldest_file)


class DirectoryEventHandler(pyinotify.ProcessEvent):
    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor
    
    def process_IN_CREATE(self, event):
        print(f"File created: {event.pathname}")
        if event.pathname.endswith('.json'):
            self.handle_event(event)
    
    def process_IN_MODIFY(self, event):
        if event.pathname.endswith('.json'):
            print(f"File modified: {event.pathname}")
            self.handle_event(event)
    
    def process_IN_MOVED_TO(self, event):
        print(f"File moved to directory: {event.pathname}")
        if event.pathname.endswith('.json'):
            self.handle_event(event)
            
    def handle_event(self, event):
        """Handle the event by listing directory and processing oldest JSON."""
        # Small delay to allow file operations to complete
        time.sleep(0.5)
        self.monitor.list_directory()
        self.monitor.process_oldest_json()

def main():
    parser = argparse.ArgumentParser(description='Monitor a directory for changes and process the oldest JSON file')
    parser.add_argument('--directory', type=str, help='Directory path to monitor')
    parser.add_argument('--destination', type=str, help='Directory path to move processed files to')
    parser.add_argument("--model-id", type=str, default="DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--chromadb-host", required=True, type=str, help="hostname or ip address where chromadb is listening")
    parser.add_argument("--chromadb-port", required=True, type=str, help="port at which chromadb is listening")
    parser.add_argument("--chromadb-collection", required=True, type=str, help="collection name in chromadb")
    parser.add_argument("--local-dir", required=True, type=str, help="local directory where index is stored")
    parser.add_argument("--llm-host", required=True, type=str, help="hostname or ip address where openai compatible llm is listening")
    parser.add_argument("--llm-port", required=True, type=str, help="port at which openai compatible llm is listening")
    parser.add_argument("--logs", required=True, type=str, help="directory where logs will be written")
    args = parser.parse_args()
    
    try:
        monitor = DirectoryMonitor(args.directory, args.destination, args.model_id,
                                    args.chromadb_host, args.chromadb_port, args.chromadb_collection,
                                    args.local_dir, args.llm_host, args.llm_port, args.logs)
        monitor.start_monitoring()
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
