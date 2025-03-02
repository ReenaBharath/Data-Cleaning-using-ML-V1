import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import psutil
import threading

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    timestamp: datetime
    function_name: str
    args_hash: str
    gpu_usage: Optional[float] = None
    batch_size: Optional[int] = None
    error_count: int = 0
    success_count: int = 0
    cache_hits: int = 0
    peak_memory: float = 0.0

class PerformanceMonitor:
    """Monitor and optimize performance of data cleaning operations."""
    
    def __init__(self, alert_thresholds: Optional[Dict[str, float]] = None):
        """Initialize performance monitor."""
        self.metrics: List[PerformanceMetrics] = []
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread = None
        
        # Set default thresholds
        self.alert_thresholds = {
            'cpu': 80,  # percentage
            'memory': 75,  # percentage
            'disk': 90,  # percentage
            'gpu_memory': 80  # percentage if GPU available
        }
        if alert_thresholds:
            self.alert_thresholds.update(alert_thresholds)
            
        # Initialize GPU monitoring if available
        self.has_gpu = False
        try:
            import torch
            self.has_gpu = torch.cuda.is_available()
        except ImportError:
            pass
            
        # Start resource monitoring
        self.start_monitoring()
    
    def add_metrics(self, metrics: PerformanceMetrics):
        """Add performance metrics to history."""
        with self._lock:
            self.metrics.append(metrics)
            # Keep only last 1000 metrics
            if len(self.metrics) > 1000:
                self.metrics = self.metrics[-1000:]
            self._check_thresholds(metrics)
    
    def _check_thresholds(self, metrics: PerformanceMetrics):
        """Check if metrics exceed thresholds and log warnings."""
        if metrics.cpu_usage > self.alert_thresholds['cpu']:
            logger.warning(f"High CPU usage detected: {metrics.cpu_usage:.1f}%")
            
        if metrics.memory_usage > self.alert_thresholds['memory']:
            logger.warning(f"High memory usage detected: {metrics.memory_usage:.1f}%")
            
        if metrics.gpu_usage and metrics.gpu_usage > self.alert_thresholds['gpu_memory']:
            logger.warning(f"High GPU memory usage detected: {metrics.gpu_usage:.1f}MB")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        with self._lock:
            if not self.metrics:
                return {}
                
            summary = {
                'total_executions': len(self.metrics),
                'total_errors': sum(m.error_count for m in self.metrics),
                'total_successes': sum(m.success_count for m in self.metrics),
                'avg_execution_time': sum(m.execution_time for m in self.metrics) / len(self.metrics),
                'avg_memory_usage': sum(m.memory_usage for m in self.metrics) / len(self.metrics),
                'avg_cpu_usage': sum(m.cpu_usage for m in self.metrics) / len(self.metrics),
                'peak_memory': max(m.peak_memory for m in self.metrics),
                'total_cache_hits': sum(m.cache_hits for m in self.metrics),
                'function_stats': self._get_function_stats()
            }
            
            if self.has_gpu:
                gpu_metrics = [m.gpu_usage for m in self.metrics if m.gpu_usage is not None]
                if gpu_metrics:
                    summary['avg_gpu_usage'] = sum(gpu_metrics) / len(gpu_metrics)
                    
            return summary
    
    def _get_function_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics per function."""
        stats = {}
        for metric in self.metrics:
            if metric.function_name not in stats:
                stats[metric.function_name] = {
                    'count': 0,
                    'avg_time': 0,
                    'error_rate': 0,
                    'avg_memory': 0
                }
            
            s = stats[metric.function_name]
            s['count'] += 1
            s['avg_time'] = (s['avg_time'] * (s['count'] - 1) + metric.execution_time) / s['count']
            s['error_rate'] = sum(1 for m in self.metrics 
                                if m.function_name == metric.function_name and m.error_count > 0) / s['count']
            s['avg_memory'] = (s['avg_memory'] * (s['count'] - 1) + metric.memory_usage) / s['count']
            
        return stats
    
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous resource monitoring."""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_resources,
                args=(interval,),
                daemon=True
            )
            self._monitor_thread.start()
            logger.info("Started performance monitoring")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            logger.info("Stopped performance monitoring")
            
    def _monitor_resources(self, interval: float):
        """Monitor system resources continuously."""
        while self._monitoring:
            try:
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                disk_percent = psutil.disk_usage('/').percent
                
                if cpu_percent > self.alert_thresholds['cpu']:
                    logger.warning(f"System CPU usage high: {cpu_percent:.1f}%")
                if memory_percent > self.alert_thresholds['memory']:
                    logger.warning(f"System memory usage high: {memory_percent:.1f}%")
                if disk_percent > self.alert_thresholds['disk']:
                    logger.warning(f"System disk usage high: {disk_percent:.1f}%")
                
                if self.has_gpu:
                    try:
                        import torch
                        gpu_memory = torch.cuda.memory_allocated() / 1024**2
                        if gpu_memory > self.alert_thresholds['gpu_memory']:
                            logger.warning(f"GPU memory usage high: {gpu_memory:.1f}MB")
                    except Exception:
                        pass
                        
            except Exception as e:
                logger.error(f"Error monitoring resources: {str(e)}")
                
            time.sleep(interval)

def monitor_performance(monitor: Optional[PerformanceMonitor] = None):
    """Performance monitoring decorator."""
    if monitor is None:
        monitor = PerformanceMonitor()
        
    def decorator(func):
        def wrapper(*args, **kwargs):
            metrics = None
            start_time = time.time()
            start_memory = psutil.Process().memory_percent()
            start_cpu = psutil.cpu_percent()
            gpu_usage = None
            
            # Track GPU usage if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    start_gpu = torch.cuda.memory_allocated()
            except ImportError:
                pass
            
            try:
                result = func(*args, **kwargs)
                
                # Calculate metrics
                execution_time = time.time() - start_time
                end_memory = psutil.Process().memory_percent()
                end_cpu = psutil.cpu_percent()
                memory_usage = end_memory - start_memory
                peak_memory = max(end_memory, start_memory)
                
                # Get GPU metrics if available
                try:
                    if 'torch' in sys.modules and torch.cuda.is_available():
                        end_gpu = torch.cuda.memory_allocated()
                        gpu_usage = (end_gpu - start_gpu) / 1024**2  # Convert to MB
                except Exception:
                    pass
                
                # Get batch size if present in kwargs
                batch_size = kwargs.get('batch_size', None)
                if batch_size is None and args:
                    # Try to get batch size from first arg if it's a list/array
                    try:
                        first_arg = args[0]
                        if hasattr(first_arg, '__len__'):
                            batch_size = len(first_arg)
                    except Exception:
                        pass
                
                metrics = PerformanceMetrics(
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                    cpu_usage=end_cpu - start_cpu,
                    timestamp=datetime.now(),
                    function_name=func.__name__,
                    args_hash=str(hash((args, tuple(sorted(kwargs.items()))))),
                    gpu_usage=gpu_usage,
                    batch_size=batch_size,
                    success_count=1,
                    error_count=0,
                    peak_memory=peak_memory
                )
                
                monitor.add_metrics(metrics)  # Add metrics right away
                return result
                
            except Exception as e:
                # Log error metrics
                execution_time = time.time() - start_time
                metrics = PerformanceMetrics(
                    execution_time=execution_time,
                    memory_usage=psutil.Process().memory_percent() - start_memory,
                    cpu_usage=psutil.cpu_percent() - start_cpu,
                    timestamp=datetime.now(),
                    function_name=func.__name__,
                    args_hash=str(hash((args, tuple(sorted(kwargs.items()))))),
                    gpu_usage=gpu_usage,
                    error_count=1,
                    success_count=0,
                    peak_memory=psutil.Process().memory_percent()
                )
                monitor.add_metrics(metrics)  # Add error metrics right away
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
                
        return wrapper
    return decorator

# Test functions
_global_monitor = PerformanceMonitor()

@monitor_performance(_global_monitor)
def cpu_intensive_task(size: int = 1000000):
    """Test function that performs CPU-intensive operations."""
    logger.info(f"Starting CPU-intensive task with size {size}")
    arr = np.random.rand(size)
    for _ in range(10):
        arr = np.sort(arr)
    return np.sum(arr)

@monitor_performance(_global_monitor)
def memory_intensive_task(size: int = 10000000):
    """Test function that performs memory-intensive operations."""
    logger.info(f"Starting memory-intensive task with size {size}")
    arrays = []
    for i in range(10):
        arrays.append(np.random.rand(size // 10))
        time.sleep(0.1)  # Give time for monitoring
    result = sum(np.sum(arr) for arr in arrays)
    return result

def main():
    try:
        logger.info("Starting performance monitoring tests...")
        
        # Test CPU-intensive operations
        logger.info("\nRunning CPU-intensive tasks...")
        for size in [100000, 500000, 1000000]:
            result = cpu_intensive_task(size)
            logger.info(f"CPU task result (size={size}): {result:.2f}")
            time.sleep(0.5)  # Give time for monitoring
        
        # Test memory-intensive operations
        logger.info("\nRunning memory-intensive tasks...")
        for size in [1000000, 5000000, 10000000]:
            result = memory_intensive_task(size)
            logger.info(f"Memory task result (size={size}): {result:.2f}")
            time.sleep(0.5)  # Give time for monitoring
        
        # Get and display performance summary using the global monitor
        logger.info("\nGathering performance metrics...")
        summary = _global_monitor.get_metrics_summary()
        
        logger.info("\n" + "="*50)
        logger.info("PERFORMANCE MONITORING SUMMARY")
        logger.info("="*50)
        
        logger.info("\nOverall Statistics:")
        logger.info(f"Total Executions: {summary.get('total_executions', 0)}")
        logger.info(f"Total Successes: {summary.get('total_successes', 0)}")
        logger.info(f"Total Errors: {summary.get('total_errors', 0)}")
        logger.info(f"Average Execution Time: {summary.get('avg_execution_time', 0):.3f} seconds")
        logger.info(f"Average Memory Usage: {summary.get('avg_memory_usage', 0):.2f}%")
        logger.info(f"Average CPU Usage: {summary.get('avg_cpu_usage', 0):.2f}%")
        logger.info(f"Peak Memory: {summary.get('peak_memory', 0):.2f}%")
        
        if 'avg_gpu_usage' in summary:
            logger.info(f"Average GPU Usage: {summary['avg_gpu_usage']:.2f}MB")
        
        logger.info("\nFunction Statistics:")
        for func_name, stats in summary.get('function_stats', {}).items():
            logger.info(f"\n{func_name}:")
            logger.info(f"  Count: {stats['count']}")
            logger.info(f"  Average Time: {stats['avg_time']:.3f} seconds")
            logger.info(f"  Error Rate: {stats['error_rate']:.2%}")
            logger.info(f"  Average Memory: {stats['avg_memory']:.2f}%")
        
        logger.info("\n" + "="*50)
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
