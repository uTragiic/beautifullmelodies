"""
Scheduling system for model training across different timeframes.
Handles periodic retraining, updates, and validation scheduling.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import schedule
import time
import pytz
from threading import Thread, Event
import queue
import numpy as np
from .trainer import ModelTrainer, TrainingConfig
from ..universe import UniverseManager

logger = logging.getLogger(__name__)

@dataclass
class ScheduleConfig:
    """Configuration for training schedules."""
    market_model_schedule: str = "0 0 * * 0"  # Weekly on Sunday at midnight
    sector_model_schedule: str = "0 0 * * 1"  # Weekly on Monday at midnight
    cluster_model_schedule: str = "0 0 * * *"  # Daily at midnight
    validation_schedule: str = "0 * * * *"     # Hourly
    max_training_time: int = 7200              # 2 hours max per training session
    market_hours_only: bool = True
    timezone: str = "America/New_York"
    retry_attempts: int = 3
    retry_delay: int = 300                     # 5 minutes
    emergency_retrain_threshold: float = 0.3    # 30% performance degradation

class TrainingScheduler:
    """
    Manages scheduling of model training and updates.
    """
    
    def __init__(self,
                 trainer: ModelTrainer,
                 universe_manager: UniverseManager,
                 config: Optional[ScheduleConfig] = None,
                 schedule_file: Optional[str] = None):
        """
        Initialize TrainingScheduler.
        
        Args:
            trainer: ModelTrainer instance
            universe_manager: UniverseManager instance
            config: Schedule configuration
            schedule_file: Path to custom schedule file
        """
        self.trainer = trainer
        self.universe_manager = universe_manager
        self.config = config or ScheduleConfig()
        
        # Initialize timezone
        self.tz = pytz.timezone(self.config.timezone)
        
        # Setup schedule from file if provided
        if schedule_file:
            self.load_schedule(schedule_file)
        else:
            self._setup_default_schedule()
            
        # Initialize monitoring
        self.stop_event = Event()
        self.task_queue = queue.Queue()
        self.training_status: Dict[str, Any] = {}
        self.performance_history: Dict[str, List[float]] = {}
        
    def _setup_default_schedule(self) -> None:
        """Setup default training schedule."""
        # Market model training (weekly)
        schedule.every().sunday.at("00:00").do(
            self._schedule_task,
            self._train_market_model,
            "market_model"
        )
        
        # Sector model training (weekly)
        schedule.every().monday.at("00:00").do(
            self._schedule_task,
            self._train_sector_models,
            "sector_models"
        )
        
        # Cluster model training (daily)
        schedule.every().day.at("00:00").do(
            self._schedule_task,
            self._train_cluster_models,
            "cluster_models"
        )
        
        # Model validation (hourly)
        schedule.every().hour.do(
            self._schedule_task,
            self._validate_models,
            "validation"
        )
        
    def start(self) -> None:
        """Start the training scheduler."""
        try:
            logger.info("Starting training scheduler...")
            
            # Start task processing thread
            self.task_thread = Thread(target=self._process_tasks)
            self.task_thread.daemon = True
            self.task_thread.start()
            
            # Start schedule monitoring thread
            self.schedule_thread = Thread(target=self._run_schedule)
            self.schedule_thread.daemon = True
            self.schedule_thread.start()
            
            logger.info("Scheduler started successfully")
            
        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")
            raise
            
    def stop(self) -> None:
        """Stop the training scheduler."""
        try:
            logger.info("Stopping training scheduler...")
            self.stop_event.set()
            
            # Wait for threads to complete
            self.task_thread.join(timeout=30)
            self.schedule_thread.join(timeout=30)
            
            logger.info("Scheduler stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
            raise
            
    def _run_schedule(self) -> None:
        """Run the schedule monitoring loop."""
        while not self.stop_event.is_set():
            try:
                # Check if within market hours
                if self.config.market_hours_only and not self._is_market_hours():
                    time.sleep(60)
                    continue
                    
                schedule.run_pending()
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in schedule loop: {e}")
                time.sleep(60)
                
    def _process_tasks(self) -> None:
        """Process training tasks from queue."""
        while not self.stop_event.is_set():
            try:
                task, name = self.task_queue.get(timeout=1)
                
                # Update status
                self.training_status[name] = {
                    'status': 'running',
                    'start_time': datetime.now(self.tz),
                    'attempts': 0
                }
                
                # Execute task with retry logic
                success = False
                while (not success and 
                       self.training_status[name]['attempts'] < self.config.retry_attempts):
                    try:
                        task()
                        success = True
                        self.training_status[name]['status'] = 'completed'
                    except Exception as e:
                        self.training_status[name]['attempts'] += 1
                        logger.error(f"Error in task {name}: {e}")
                        if self.training_status[name]['attempts'] < self.config.retry_attempts:
                            time.sleep(self.config.retry_delay)
                            
                if not success:
                    self.training_status[name]['status'] = 'failed'
                    self._handle_training_failure(name)
                    
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing tasks: {e}")
                time.sleep(60)
                
    def _schedule_task(self, task: callable, name: str) -> None:
        """Add task to processing queue."""
        self.task_queue.put((task, name))
        
    def _train_market_model(self) -> None:
        """Train market model task."""
        end_date = datetime.now(self.tz).strftime('%Y-%m-%d')
        start_date = (datetime.now(self.tz) - timedelta(days=365)).strftime('%Y-%m-%d')
        
        self.trainer.train_market_model(start_date, end_date)
        
    def _train_sector_models(self) -> None:
        """Train sector models task."""
        end_date = datetime.now(self.tz).strftime('%Y-%m-%d')
        start_date = (datetime.now(self.tz) - timedelta(days=365)).strftime('%Y-%m-%d')
        
        self.trainer.train_sector_models(start_date, end_date)
        
    def _train_cluster_models(self) -> None:
        """Train cluster models task."""
        end_date = datetime.now(self.tz).strftime('%Y-%m-%d')
        start_date = (datetime.now(self.tz) - timedelta(days=180)).strftime('%Y-%m-%d')
        
        self.trainer.train_cluster_models(start_date, end_date)
        
    def _validate_models(self) -> None:
        """Validate all models task."""
        try:
            # Get recent performance metrics
            metrics = self.trainer.get_performance_metrics()
            
            # Check for performance degradation
            for model_name, performance in metrics.items():
                if model_name not in self.performance_history:
                    self.performance_history[model_name] = []
                    
                self.performance_history[model_name].append(performance['sharpe_ratio'])
                
                # Check if emergency retraining is needed
                if self._check_emergency_retrain(model_name):
                    self._schedule_emergency_retrain(model_name)
                    
        except Exception as e:
            logger.error(f"Error in model validation: {e}")
            raise
            
    def _check_emergency_retrain(self, model_name: str) -> bool:
        """Check if emergency retraining is needed."""
        if len(self.performance_history[model_name]) < 2:
            return False
            
        recent_perf = self.performance_history[model_name][-1]
        baseline_perf = np.mean(self.performance_history[model_name][:-1])
        
        if baseline_perf == 0:
            return False
            
        degradation = (baseline_perf - recent_perf) / abs(baseline_perf)
        return degradation > self.config.emergency_retrain_threshold
        
    def _schedule_emergency_retrain(self, model_name: str) -> None:
        """Schedule emergency retraining for a model."""
        logger.warning(f"Scheduling emergency retraining for {model_name}")
        
        if 'market_model' in model_name:
            self._schedule_task(self._train_market_model, f"emergency_{model_name}")
        elif 'sector_model' in model_name:
            self._schedule_task(self._train_sector_models, f"emergency_{model_name}")
        elif 'cluster_model' in model_name:
            self._schedule_task(self._train_cluster_models, f"emergency_{model_name}")
            
    def _handle_training_failure(self, task_name: str) -> None:
        """Handle training task failure."""
        logger.error(f"Training task {task_name} failed after {self.config.retry_attempts} attempts")
        
        # Send alert or notification
        self._send_alert(f"Training failure: {task_name}")
        
        # Load backup model if available
        self._load_backup_model(task_name)
        
    def _is_market_hours(self) -> bool:
        """Check if currently within market hours."""
        now = datetime.now(self.tz)
        
        # Check if weekend
        if now.weekday() > 4:
            return False
            
        # Check if within trading hours (9:30 AM - 4:00 PM)
        market_open = now.replace(hour=9, minute=30)
        market_close = now.replace(hour=16, minute=0)
        
        return market_open <= now <= market_close
        
    def _send_alert(self, message: str) -> None:
        """Send alert for important events."""
        logger.critical(message)
        # Implement alert mechanism (email, Slack, etc.)
        
    def _load_backup_model(self, task_name: str) -> None:
        """Load backup model after training failure."""
        try:
            if 'market_model' in task_name:
                self.trainer.load_market_model_backup()
            elif 'sector_model' in task_name:
                self.trainer.load_sector_models_backup()
            elif 'cluster_model' in task_name:
                self.trainer.load_cluster_models_backup()
                
        except Exception as e:
            logger.error(f"Error loading backup model: {e}")
            
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return self.training_status
        
    def get_next_training_times(self) -> Dict[str, datetime]:
        """Get next scheduled training times."""
        return {
            'market_model': schedule.next_run('market_model'),
            'sector_models': schedule.next_run('sector_models'),
            'cluster_models': schedule.next_run('cluster_models'),
            'validation': schedule.next_run('validation')
        }
        
    def load_schedule(self, schedule_file: str) -> None:
        """Load custom schedule from file."""
        try:
            with open(schedule_file) as f:
                schedule_data = json.load(f)
                
            schedule.clear()
            
            for task_name, task_schedule in schedule_data.items():
                if hasattr(self, f"_train_{task_name}"):
                    task_func = getattr(self, f"_train_{task_name}")
                    schedule.every().day.at(task_schedule).do(
                        self._schedule_task,
                        task_func,
                        task_name
                    )
                    
        except Exception as e:
            logger.error(f"Error loading schedule: {e}")
            self._setup_default_schedule()