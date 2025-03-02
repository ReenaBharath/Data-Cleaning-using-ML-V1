import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from src.utils.performance_monitor import PerformanceMonitor, monitor_performance
from src.utils.logging_config import setup_logging, LOGGING_CONFIG
import logging

# Setup logging
setup_logging(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Initialize performance monitor with custom thresholds
monitor = PerformanceMonitor(alert_thresholds={
    'cpu': 80,  # Lower threshold for testing
    'memory': 75,
    'disk': 90,
    'gpu_memory': 80
})

@monitor_performance(monitor=monitor)
def cpu_intensive_task(size: int = 1000000):
    """Test function that performs CPU-intensive operations."""
    logger.info(f"Starting CPU-intensive task with size {size}")
    arr = np.random.rand(size)
    for _ in range(10):
        arr = np.sort(arr)
    return np.sum(arr)

@monitor_performance(monitor=monitor)
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
        
        # Get and display performance summary
        logger.info("\nGathering performance metrics...")
        summary = monitor.get_metrics_summary()
        
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
    finally:
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()
