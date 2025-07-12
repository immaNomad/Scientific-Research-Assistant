"""
Optimization Configuration Management
Centralized configuration for performance tuning and feature toggles
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class SearchConfig:
    """Configuration for search engine optimization"""
    # Cache settings
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    max_cache_size: int = 1000
    
    # Search performance
    max_results_per_query: int = 50
    search_timeout_seconds: int = 30
    parallel_search_enabled: bool = True
    
    # Relevance scoring
    title_weight: float = 3.0
    abstract_weight: float = 1.5
    category_weight: float = 2.0
    keyword_weight: float = 2.0
    exact_match_bonus: float = 5.0
    
    # Indexing
    min_word_length: int = 3
    enable_stemming: bool = False
    enable_synonym_expansion: bool = False

@dataclass
class DatabaseConfig:
    """Configuration for database optimization"""
    # Connection settings
    connection_pool_size: int = 10
    connection_timeout_seconds: int = 30
    
    # Performance tuning
    enable_wal_mode: bool = True  # Write-Ahead Logging
    cache_size_mb: int = 64
    temp_store_memory: bool = True
    
    # Maintenance
    auto_vacuum: bool = True
    analyze_frequency_hours: int = 24
    backup_frequency_hours: int = 168  # Weekly

@dataclass
class LLMConfig:
    """Configuration for LLM optimization"""
    # Model selection
    preferred_model: str = "gemini"
    fallback_enabled: bool = True
    
    # Generation settings
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout_seconds: int = 30
    
    # Optimization
    batch_requests: bool = False
    request_pooling: bool = True
    
    # Rate limiting
    requests_per_minute: int = 60
    concurrent_requests: int = 5

@dataclass
class SystemConfig:
    """Configuration for system-level optimizations"""
    # Resource limits
    max_cpu_percent: float = 80.0
    max_memory_percent: float = 80.0
    max_disk_usage_percent: float = 90.0
    
    # Monitoring
    performance_monitoring: bool = True
    detailed_logging: bool = False
    metrics_retention_days: int = 30
    
    # Features
    enable_async_processing: bool = True
    enable_background_tasks: bool = True
    enable_health_checks: bool = True

@dataclass
class OptimizationConfig:
    """Complete optimization configuration"""
    search: SearchConfig
    database: DatabaseConfig
    llm: LLMConfig
    system: SystemConfig
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values"""
        # Validate search config
        if self.search.cache_ttl_seconds < 0:
            raise ValueError("Cache TTL cannot be negative")
        
        if self.search.max_results_per_query < 1:
            raise ValueError("Max results per query must be at least 1")
        
        # Validate database config
        if self.database.connection_pool_size < 1:
            raise ValueError("Connection pool size must be at least 1")
        
        # Validate LLM config
        if self.llm.max_tokens < 1:
            raise ValueError("Max tokens must be at least 1")
        
        if not 0 <= self.llm.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        
        # Validate system config
        if not 0 <= self.system.max_cpu_percent <= 100:
            raise ValueError("Max CPU percent must be between 0 and 100")

class ConfigManager:
    """Manages optimization configuration"""
    
    def __init__(self, config_path: str = "config/optimization.yaml"):
        self.config_path = Path(config_path)
        self.config_dir = self.config_path.parent
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self._config: Optional[OptimizationConfig] = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file or create default"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = yaml.safe_load(f)
                
                # Convert nested dicts to dataclasses
                search_config = SearchConfig(**data.get('search', {}))
                database_config = DatabaseConfig(**data.get('database', {}))
                llm_config = LLMConfig(**data.get('llm', {}))
                system_config = SystemConfig(**data.get('system', {}))
                
                self._config = OptimizationConfig(
                    search=search_config,
                    database=database_config,
                    llm=llm_config,
                    system=system_config
                )
                
                logger.info(f"Loaded configuration from {self.config_path}")
                
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                self._create_default_config()
        else:
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration"""
        self._config = OptimizationConfig(
            search=SearchConfig(),
            database=DatabaseConfig(),
            llm=LLMConfig(),
            system=SystemConfig()
        )
        
        # Save default config
        self.save_config()
        logger.info(f"Created default configuration at {self.config_path}")
    
    def save_config(self):
        """Save current configuration to file"""
        if not self._config:
            return
        
        # Convert to dict for serialization
        config_dict = {
            'search': asdict(self._config.search),
            'database': asdict(self._config.database),
            'llm': asdict(self._config.llm),
            'system': asdict(self._config.system)
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved configuration to {self.config_path}")
    
    @property
    def config(self) -> OptimizationConfig:
        """Get current configuration"""
        return self._config
    
    def update_search_config(self, **kwargs):
        """Update search configuration"""
        for key, value in kwargs.items():
            if hasattr(self._config.search, key):
                setattr(self._config.search, key, value)
        self.save_config()
    
    def update_database_config(self, **kwargs):
        """Update database configuration"""
        for key, value in kwargs.items():
            if hasattr(self._config.database, key):
                setattr(self._config.database, key, value)
        self.save_config()
    
    def update_llm_config(self, **kwargs):
        """Update LLM configuration"""
        for key, value in kwargs.items():
            if hasattr(self._config.llm, key):
                setattr(self._config.llm, key, value)
        self.save_config()
    
    def update_system_config(self, **kwargs):
        """Update system configuration"""
        for key, value in kwargs.items():
            if hasattr(self._config.system, key):
                setattr(self._config.system, key, value)
        self.save_config()
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current config"""
        recommendations = []
        
        # Check search config
        if not self._config.search.cache_enabled:
            recommendations.append("Enable search caching for better performance")
        
        if self._config.search.max_results_per_query > 100:
            recommendations.append("Consider reducing max results per query for faster responses")
        
        # Check database config
        if not self._config.database.enable_wal_mode:
            recommendations.append("Enable WAL mode for better database performance")
        
        if self._config.database.cache_size_mb < 32:
            recommendations.append("Consider increasing database cache size")
        
        # Check LLM config
        if not self._config.llm.fallback_enabled:
            recommendations.append("Enable LLM fallback for better reliability")
        
        if self._config.llm.max_tokens > 2000:
            recommendations.append("Consider reducing max tokens for faster LLM responses")
        
        # Check system config
        if not self._config.system.performance_monitoring:
            recommendations.append("Enable performance monitoring for better insights")
        
        if self._config.system.max_cpu_percent > 90:
            recommendations.append("Consider lowering max CPU threshold")
        
        return recommendations
    
    def apply_performance_profile(self, profile: str):
        """Apply a predefined performance profile"""
        if profile == "fast":
            # Optimize for speed
            self.update_search_config(
                cache_enabled=True,
                max_results_per_query=20,
                parallel_search_enabled=True
            )
            self.update_llm_config(
                max_tokens=500,
                temperature=0.3,
                concurrent_requests=10
            )
            
        elif profile == "balanced":
            # Balanced performance and quality
            self.update_search_config(
                cache_enabled=True,
                max_results_per_query=50,
                parallel_search_enabled=True
            )
            self.update_llm_config(
                max_tokens=1000,
                temperature=0.7,
                concurrent_requests=5
            )
            
        elif profile == "quality":
            # Optimize for quality
            self.update_search_config(
                cache_enabled=True,
                max_results_per_query=100,
                enable_synonym_expansion=True
            )
            self.update_llm_config(
                max_tokens=2000,
                temperature=0.7,
                concurrent_requests=3
            )
            
        elif profile == "memory_optimized":
            # Optimize for low memory usage
            self.update_search_config(
                max_cache_size=100,
                max_results_per_query=25
            )
            self.update_database_config(
                cache_size_mb=32,
                connection_pool_size=5
            )
            
        else:
            raise ValueError(f"Unknown performance profile: {profile}")
        
        logger.info(f"Applied performance profile: {profile}")
    
    def export_config(self, format: str = "yaml") -> str:
        """Export configuration as string"""
        if format == "yaml":
            config_dict = {
                'search': asdict(self._config.search),
                'database': asdict(self._config.database),
                'llm': asdict(self._config.llm),
                'system': asdict(self._config.system)
            }
            return yaml.dump(config_dict, default_flow_style=False, indent=2)
        
        elif format == "json":
            config_dict = {
                'search': asdict(self._config.search),
                'database': asdict(self._config.database),
                'llm': asdict(self._config.llm),
                'system': asdict(self._config.system)
            }
            return json.dumps(config_dict, indent=2)
        
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self._config = OptimizationConfig(
            search=SearchConfig(),
            database=DatabaseConfig(),
            llm=LLMConfig(),
            system=SystemConfig()
        )
        self.save_config()
        logger.info("Reset configuration to defaults")

# Global config manager
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get the global configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config() -> OptimizationConfig:
    """Get the current optimization configuration"""
    return get_config_manager().config 
