"""
Performance Monitoring and Analytics System
Tracks usage patterns, performance metrics, and system health
"""

import asyncio
import time
import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import psutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class QueryMetrics:
    """Metrics for a single query"""
    query: str
    timestamp: float
    response_time: float
    results_count: int
    search_type: str  # 'enhanced', 'basic', 'cached'
    domain: str
    success: bool
    error_message: Optional[str] = None

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    active_searches: int
    cache_hit_rate: float
    database_size_mb: float

@dataclass
class UsageAnalytics:
    """User behavior analytics"""
    most_common_queries: List[Tuple[str, int]]
    average_response_time: float
    peak_usage_hours: List[int]
    search_success_rate: float
    domain_preferences: Dict[str, int]

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system
    """
    
    def __init__(self, db_path: str = "data/analytics.db"):
        self.db_path = db_path
        self.metrics_queue = deque(maxlen=1000)  # Keep last 1000 queries
        self.system_metrics = deque(maxlen=100)   # Keep last 100 system snapshots
        self.lock = threading.Lock()
        
        # Performance thresholds
        self.slow_query_threshold = 5.0  # seconds
        self.high_memory_threshold = 80.0  # percent
        self.high_cpu_threshold = 80.0   # percent
        
        # Initialize database
        self._init_database()
        
        # Start background monitoring
        self._start_monitoring()
        
        logger.info("Performance monitoring initialized")
    
    def _init_database(self):
        """Initialize analytics database"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Query metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    response_time REAL NOT NULL,
                    results_count INTEGER NOT NULL,
                    search_type TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT
                )
            """)
            
            # System metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    cpu_percent REAL NOT NULL,
                    memory_percent REAL NOT NULL,
                    disk_usage_percent REAL NOT NULL,
                    active_searches INTEGER NOT NULL,
                    cache_hit_rate REAL NOT NULL,
                    database_size_mb REAL NOT NULL
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query_timestamp ON query_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_system_timestamp ON system_metrics(timestamp)")
            conn.commit()
    
    def _start_monitoring(self):
        """Start background system monitoring"""
        def monitor_system():
            while True:
                try:
                    self._collect_system_metrics()
                    time.sleep(60)  # Collect system metrics every minute
                except Exception as e:
                    logger.error(f"Error collecting system metrics: {e}")
                    time.sleep(60)
        
        thread = threading.Thread(target=monitor_system, daemon=True)
        thread.start()
    
    def _collect_system_metrics(self):
        """Collect current system metrics"""
        try:
            # Get system info
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get database size
            db_size_mb = 0
            if os.path.exists("data/papers/papers.db"):
                db_size_mb = os.path.getsize("data/papers/papers.db") / (1024 * 1024)
            
            # Calculate cache hit rate from recent queries
            cache_hit_rate = self._calculate_cache_hit_rate()
            
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage_percent=disk.percent,
                active_searches=0,  # Would need to track active searches
                cache_hit_rate=cache_hit_rate,
                database_size_mb=db_size_mb
            )
            
            # Store in memory and database
            with self.lock:
                self.system_metrics.append(metrics)
            
            self._store_system_metrics(metrics)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate from recent queries"""
        with self.lock:
            if not self.metrics_queue:
                return 0.0
            
            recent_queries = [q for q in self.metrics_queue if q.timestamp > time.time() - 3600]  # Last hour
            if not recent_queries:
                return 0.0
            
            cache_hits = sum(1 for q in recent_queries if q.search_type == 'cached')
            return (cache_hits / len(recent_queries)) * 100
    
    def log_query(self, query: str, response_time: float, results_count: int, 
                  search_type: str, domain: str, success: bool, error_message: str = None):
        """Log a query with its metrics"""
        metrics = QueryMetrics(
            query=query,
            timestamp=time.time(),
            response_time=response_time,
            results_count=results_count,
            search_type=search_type,
            domain=domain,
            success=success,
            error_message=error_message
        )
        
        # Store in memory
        with self.lock:
            self.metrics_queue.append(metrics)
        
        # Store in database
        self._store_query_metrics(metrics)
        
        # Check for performance issues
        self._check_performance_alerts(metrics)
    
    def _store_query_metrics(self, metrics: QueryMetrics):
        """Store query metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO query_metrics 
                    (query, timestamp, response_time, results_count, search_type, domain, success, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.query,
                    metrics.timestamp,
                    metrics.response_time,
                    metrics.results_count,
                    metrics.search_type,
                    metrics.domain,
                    metrics.success,
                    metrics.error_message
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing query metrics: {e}")
    
    def _store_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO system_metrics 
                    (timestamp, cpu_percent, memory_percent, disk_usage_percent, 
                     active_searches, cache_hit_rate, database_size_mb)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.timestamp,
                    metrics.cpu_percent,
                    metrics.memory_percent,
                    metrics.disk_usage_percent,
                    metrics.active_searches,
                    metrics.cache_hit_rate,
                    metrics.database_size_mb
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing system metrics: {e}")
    
    def _check_performance_alerts(self, metrics: QueryMetrics):
        """Check for performance issues and log alerts"""
        if metrics.response_time > self.slow_query_threshold:
            logger.warning(f"Slow query detected: '{metrics.query}' took {metrics.response_time:.2f}s")
        
        if not metrics.success:
            logger.error(f"Query failed: '{metrics.query}' - {metrics.error_message}")
    
    def get_usage_analytics(self, hours: int = 24) -> UsageAnalytics:
        """Get usage analytics for the specified time period"""
        cutoff_time = time.time() - (hours * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get queries from the specified time period
            cursor = conn.execute("""
                SELECT * FROM query_metrics 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            """, (cutoff_time,))
            
            queries = [dict(row) for row in cursor.fetchall()]
        
        if not queries:
            return UsageAnalytics(
                most_common_queries=[],
                average_response_time=0.0,
                peak_usage_hours=[],
                search_success_rate=0.0,
                domain_preferences={}
            )
        
        # Analyze data
        query_counts = defaultdict(int)
        domain_counts = defaultdict(int)
        hour_counts = defaultdict(int)
        response_times = []
        successful_queries = 0
        
        for query_data in queries:
            query_counts[query_data['query']] += 1
            domain_counts[query_data['domain']] += 1
            
            # Convert timestamp to hour
            hour = datetime.fromtimestamp(query_data['timestamp']).hour
            hour_counts[hour] += 1
            
            response_times.append(query_data['response_time'])
            if query_data['success']:
                successful_queries += 1
        
        # Calculate analytics
        most_common_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        average_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        peak_usage_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        peak_usage_hours = [hour for hour, count in peak_usage_hours]
        search_success_rate = (successful_queries / len(queries)) * 100 if queries else 0.0
        
        return UsageAnalytics(
            most_common_queries=most_common_queries,
            average_response_time=average_response_time,
            peak_usage_hours=peak_usage_hours,
            search_success_rate=search_success_rate,
            domain_preferences=dict(domain_counts)
        )
    
    def get_performance_summary(self) -> Dict:
        """Get current performance summary"""
        with self.lock:
            recent_queries = [q for q in self.metrics_queue if q.timestamp > time.time() - 3600]
            latest_system = self.system_metrics[-1] if self.system_metrics else None
        
        if not recent_queries:
            return {
                'status': 'no_data',
                'message': 'No recent queries to analyze'
            }
        
        # Calculate performance metrics
        avg_response_time = sum(q.response_time for q in recent_queries) / len(recent_queries)
        slow_queries = [q for q in recent_queries if q.response_time > self.slow_query_threshold]
        failed_queries = [q for q in recent_queries if not q.success]
        
        status = 'healthy'
        warnings = []
        
        if avg_response_time > 3.0:
            status = 'degraded'
            warnings.append(f"Average response time high: {avg_response_time:.2f}s")
        
        if len(slow_queries) > len(recent_queries) * 0.1:  # More than 10% slow queries
            status = 'degraded'
            warnings.append(f"Many slow queries: {len(slow_queries)}/{len(recent_queries)}")
        
        if len(failed_queries) > len(recent_queries) * 0.05:  # More than 5% failed queries
            status = 'degraded'
            warnings.append(f"Query failures: {len(failed_queries)}/{len(recent_queries)}")
        
        if latest_system:
            if latest_system.cpu_percent > self.high_cpu_threshold:
                status = 'degraded'
                warnings.append(f"High CPU usage: {latest_system.cpu_percent:.1f}%")
            
            if latest_system.memory_percent > self.high_memory_threshold:
                status = 'degraded'
                warnings.append(f"High memory usage: {latest_system.memory_percent:.1f}%")
        
        return {
            'status': status,
            'warnings': warnings,
            'metrics': {
                'queries_last_hour': len(recent_queries),
                'average_response_time': avg_response_time,
                'success_rate': (len(recent_queries) - len(failed_queries)) / len(recent_queries) * 100,
                'slow_queries': len(slow_queries),
                'cache_hit_rate': self._calculate_cache_hit_rate(),
                'system_metrics': asdict(latest_system) if latest_system else None
            }
        }
    
    def generate_report(self, hours: int = 24) -> str:
        """Generate a comprehensive performance report"""
        analytics = self.get_usage_analytics(hours)
        performance = self.get_performance_summary()
        
        report = f"""
🔍 Research Assistant Performance Report
{'=' * 50}

📊 Usage Analytics ({hours} hours)
• Total queries: {len(analytics.most_common_queries)}
• Average response time: {analytics.average_response_time:.2f} seconds
• Search success rate: {analytics.search_success_rate:.1f}%
• Cache hit rate: {performance['metrics']['cache_hit_rate']:.1f}%

🔥 Most Common Queries:
"""
        
        for query, count in analytics.most_common_queries[:5]:
            report += f"   • {query} ({count} times)\n"
        
        report += f"""
🕐 Peak Usage Hours: {', '.join(map(str, analytics.peak_usage_hours))}

📈 Domain Preferences:
"""
        
        for domain, count in analytics.domain_preferences.items():
            report += f"   • {domain}: {count} queries\n"
        
        report += f"""
⚡ System Performance:
• Status: {performance['status'].upper()}
• Queries last hour: {performance['metrics']['queries_last_hour']}
• Current response time: {performance['metrics']['average_response_time']:.2f}s
• Success rate: {performance['metrics']['success_rate']:.1f}%
"""
        
        if performance['warnings']:
            report += f"\n⚠️ Warnings:\n"
            for warning in performance['warnings']:
                report += f"   • {warning}\n"
        
        return report
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old analytics data"""
        cutoff_time = time.time() - (days * 24 * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM query_metrics WHERE timestamp < ?", (cutoff_time,))
            conn.execute("DELETE FROM system_metrics WHERE timestamp < ?", (cutoff_time,))
            conn.commit()
        
        logger.info(f"Cleaned up analytics data older than {days} days")

# Global performance monitor instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

def log_query_performance(query: str, response_time: float, results_count: int, 
                         search_type: str, domain: str, success: bool, error_message: str = None):
    """Convenience function to log query performance"""
    monitor = get_performance_monitor()
    monitor.log_query(query, response_time, results_count, search_type, domain, success, error_message) 