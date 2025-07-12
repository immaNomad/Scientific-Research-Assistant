#!/usr/bin/env python3
"""Performance monitoring script"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    from analytics.performance_monitor import get_performance_monitor
    
    monitor = get_performance_monitor()
    
    print("📊 Performance Monitor")
    print("=" * 40)
    
    # Show performance summary
    summary = monitor.get_performance_summary()
    print(f"Status: {summary['status'].upper()}")
    
    if summary['warnings']:
        print("\n⚠️  Warnings:")
        for warning in summary['warnings']:
            print(f"   • {warning}")
    
    # Show usage analytics
    analytics = monitor.get_usage_analytics(24)  # Last 24 hours
    print(f"\n📈 Usage Analytics (24h):")
    print(f"   Average response time: {analytics.average_response_time:.2f}s")
    print(f"   Success rate: {analytics.search_success_rate:.1f}%")
    
    if analytics.most_common_queries:
        print(f"\n🔥 Most Common Queries:")
        for query, count in analytics.most_common_queries[:5]:
            print(f"   • {query} ({count} times)")
    
    # Generate full report
    print("\n" + "=" * 40)
    print(monitor.generate_report(24))

if __name__ == "__main__":
    main()
