import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import logging
from src.utils.performance_monitor import PerformanceMonitor, monitor_performance
from src.utils.logging_config import setup_logging, LOGGING_CONFIG

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
def process_large_array(size: int = 1000000):
    """Test function that processes a large array."""
    logger.info(f"Processing array of size {size}")
    arr = np.random.rand(size)
    time.sleep(0.1)  # Simulate some processing
    return np.sum(arr)

@monitor_performance(monitor=monitor)
def process_batch(batch_data):
    """Test function that processes a batch of data."""
    logger.info(f"Processing batch of size {len(batch_data)}")
    time.sleep(0.05)  # Simulate batch processing
    return [x * 2 for x in batch_data]

def print_summary(summary):
    """Pretty print the performance summary."""
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

def main():
    logger.info("Starting performance monitoring test...")
    
    try:
        # Test 1: Process large array multiple times
        logger.info("\nTest 1: Processing large arrays...")
        for i in range(3):
            result = process_large_array()
            logger.info(f"Array sum {i+1}: {result:.2f}")
        
        # Test 2: Batch processing
        logger.info("\nTest 2: Batch processing...")
        batches = [list(range(i, i+5)) for i in range(0, 15, 5)]
        for batch in batches:
            result = process_batch(batch)
            logger.info(f"Processed batch: {result}")
        
        # Get and print performance summary
        summary = monitor.get_metrics_summary()
        print_summary(summary)
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise
    finally:
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()
