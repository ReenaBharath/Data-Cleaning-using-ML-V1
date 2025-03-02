"""Unit tests for performance monitoring utilities."""

import pytest
import time
import psutil
from src.utils.performance_monitor import PerformanceMonitor, PerformanceMetrics

@pytest.fixture
def monitor():
    """Create a performance monitor instance."""
    return PerformanceMonitor(
        alert_thresholds={
            'cpu': 90,
            'memory': 85,
            'disk': 95,
            'gpu_memory': 90
        }
    )

class TestPerformanceMonitor:
    """Test performance monitoring functionality."""
    
    def test_monitor_initialization(self, monitor):
        assert monitor.alert_thresholds['cpu'] == 90
        assert monitor.alert_thresholds['memory'] == 85
        assert monitor._monitoring is False
        assert monitor._monitor_thread is None
        
    def test_start_stop_monitoring(self, monitor):
        monitor.start_monitoring(interval=0.1)
        assert monitor._monitoring is True
        assert monitor._monitor_thread is not None
        assert monitor._monitor_thread.is_alive()
        
        time.sleep(0.2)  # Allow some monitoring to occur
        
        monitor.stop_monitoring()
        assert monitor._monitoring is False
        assert not monitor._monitor_thread.is_alive()
        
    def test_optimize_resources(self, monitor):
        # Create some garbage to clean up
        large_list = [i for i in range(1000000)]
        initial_memory = psutil.Process().memory_percent()
        
        monitor.optimize_resources()
        
        # Memory should be reduced after optimization
        final_memory = psutil.Process().memory_percent()
        assert final_memory <= initial_memory
        
        # Clear reference to force garbage collection
        del large_list
        
    def test_add_metrics(self, monitor):
        metrics = PerformanceMetrics(
            function_name="test_func",
            execution_time=1.5,
            memory_usage=50.0,
            cpu_usage=30.0,
            timestamp=time.time()
        )
        
        monitor.add_metrics(metrics)
        assert len(monitor.metrics) == 1
        assert monitor.metrics[0].function_name == "test_func"
        
    def test_get_detailed_metrics(self, monitor):
        # Add multiple metrics
        for i in range(3):
            metrics = PerformanceMetrics(
                function_name=f"func_{i}",
                execution_time=1.0 * i,
                memory_usage=50.0,
                cpu_usage=30.0,
                timestamp=time.time()
            )
            monitor.add_metrics(metrics)
            
        detailed_metrics = monitor.get_detailed_metrics()
        
        assert 'overall' in detailed_metrics
        assert 'by_function' in detailed_metrics
        assert len(detailed_metrics['by_function']) == 3
        
        # Check overall metrics
        assert detailed_metrics['overall']['metrics_count'] == 3
        assert 'avg_execution_time' in detailed_metrics['overall']
        assert 'avg_memory_usage' in detailed_metrics['overall']
        assert 'avg_cpu_usage' in detailed_metrics['overall']
        
        # Check function-specific metrics
        for i in range(3):
            func_name = f"func_{i}"
            assert func_name in detailed_metrics['by_function']
            func_metrics = detailed_metrics['by_function'][func_name]
            assert 'count' in func_metrics
            assert 'avg_execution_time' in func_metrics
            assert 'avg_memory_usage' in func_metrics
            assert 'avg_cpu_usage' in func_metrics
            assert 'last_execution' in func_metrics
            
    def test_monitor_high_resource_usage(self, monitor, caplog):
        # Mock high resource usage
        def mock_high_cpu():
            return 95.0
            
        def mock_high_memory():
            return type('obj', (object,), {'percent': 90.0})
            
        original_cpu = psutil.cpu_percent
        original_virtual_memory = psutil.virtual_memory
        
        try:
            psutil.cpu_percent = mock_high_cpu
            psutil.virtual_memory = mock_high_memory
            
            monitor.start_monitoring(interval=0.1)
            time.sleep(0.2)  # Allow monitoring to detect high usage
            monitor.stop_monitoring()
            
            # Check if warnings were logged
            assert any("High CPU usage" in record.message 
                     for record in caplog.records)
            assert any("High memory usage" in record.message 
                     for record in caplog.records)
            
        finally:
            # Restore original functions
            psutil.cpu_percent = original_cpu
            psutil.virtual_memory = original_virtual_memory
            
    @pytest.mark.skipif(not torch.cuda.is_available(), 
                       reason="GPU not available")
    def test_gpu_monitoring(self, monitor):
        monitor.start_monitoring(interval=0.1)
        time.sleep(0.2)
        monitor.stop_monitoring()
        
        # If GPU is available, has_gpu should be True
        assert monitor.has_gpu is True
