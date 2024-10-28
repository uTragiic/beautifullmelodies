"""
Configuration management system for the trading platform.
Provides centralized configuration with validation, environment handling,
and runtime updates.
"""

# Standard Library Imports
import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Third-Party Imports
import yaml

# Local Imports
from ..utils.validation import validate_parameters, ValidationError

logger = logging.getLogger(__name__)

class ConfigurationManager:
    """
    Manages trading system configuration with validation and environment handling.
    """
    
    def __init__(self, config_path: str, env: str = "development"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration files
            env: Environment name (development, production, test)
        """
        self.config_path = Path(config_path)
        self.env = env
        self.config: Dict[str, Any] = {}
        self.base_config: Dict[str, Any] = {}
        
        # Define required configuration parameters and their types
        self.required_params = {
            'database': {
                'host': str,
                'port': int,
                'name': str
            },
            'trading': {
                'max_position_size': float,
                'max_drawdown': float,
                'risk_free_rate': float
            },
            'risk': {
                'max_risk_per_trade': float,
                'max_correlation': float,
                'max_portfolio_heat': float
            },
            'signal': {
                'confidence_threshold': float,
                'minimum_signal_strength': float
            }
        }
        
        # Load configuration
        self._load_config()
        
    def _load_config(self) -> None:
        """Load and validate configuration files."""
        try:
            # Load base configuration
            base_config_path = self.config_path / 'base.yaml'
            if base_config_path.exists():
                with open(base_config_path) as f:
                    self.base_config = yaml.safe_load(f)
            
            # Load environment-specific configuration
            env_config_path = self.config_path / f'{self.env}.yaml'
            env_config = {}
            if env_config_path.exists():
                with open(env_config_path) as f:
                    env_config = yaml.safe_load(f)
            
            # Merge configurations
            self.config = self._merge_configs(self.base_config, env_config)
            
            # Override with environment variables
            self._apply_env_overrides()
            
            # Validate configuration
            self._validate_config()
            
            logger.info(f"Configuration loaded for environment: {self.env}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
            
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge configuration dictionaries.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration dictionary
        """
        merged = base.copy()
        
        for key, value in override.items():
            if (
                key in merged and 
                isinstance(merged[key], dict) and 
                isinstance(value, dict)
            ):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
                
        return merged
        
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        prefix = "TRADING_"
        
        for key in os.environ:
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                value = os.environ[key]
                
                # Convert value to appropriate type
                try:
                    if value.lower() in ('true', 'false'):
                        value = value.lower() == 'true'
                    elif value.isdigit():
                        value = int(value)
                    elif '.' in value and all(p.isdigit() for p in value.split('.')):
                        value = float(value)
                except ValueError:
                    pass  # Keep as string if conversion fails
                
                # Update nested configuration
                self._update_nested_config(config_key.split('_'), value)
                
    def _update_nested_config(self, keys: list, value: Any) -> None:
        """
        Update nested configuration dictionary.
        
        Args:
            keys: List of nested keys
            value: Value to set
        """
        current = self.config
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value
        
    def _validate_config(self) -> None:
        """Validate configuration against required parameters."""
        try:
            self._validate_section(self.config, self.required_params)
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
            
    def _validate_section(self, config: Dict[str, Any], required: Dict[str, Any]) -> None:
        """
        Recursively validate configuration section.
        
        Args:
            config: Configuration section to validate
            required: Required parameters for section
        """
        for key, value in required.items():
            if key not in config:
                raise ValidationError(f"Missing required configuration parameter: {key}")
                
            if isinstance(value, dict):
                if not isinstance(config[key], dict):
                    raise ValidationError(
                        f"Configuration section {key} must be a dictionary"
                    )
                self._validate_section(config[key], value)
            else:
                if not isinstance(config[key], value):
                    raise ValidationError(
                        f"Configuration parameter {key} must be of type {value.__name__}"
                    )
                    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (dot-separated for nested keys)
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value
        """
        try:
            value = self.config
            for k in key.split('.'):
                value = value[k]
            return value
        except KeyError:
            return default
            
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (dot-separated for nested keys)
            value: Value to set
        """
        keys = key.split('.')
        current = self.config
        
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        
        current[keys[-1]] = value
        
        try:
            self._validate_config()
        except ValidationError as e:
            # Revert change if validation fails
            self._load_config()
            raise ValidationError(f"Invalid configuration update: {e}")
            
    def save(self) -> None:
        """Save current configuration to file."""
        try:
            config_file = self.config_path / f'{self.env}.yaml'
            with open(config_file, 'w') as f:
                yaml.safe_dump(self.config, f)
            logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration section."""
        return self.get('database', {})

    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration section."""
        return self.get('trading', {})

    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration section."""
        return self.get('risk', {})

    def get_signal_config(self) -> Dict[str, Any]:
        """Get signal generation configuration section."""
        return self.get('signal', {})

    def to_dict(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary."""
        return self.config.copy()

    def reset(self) -> None:
        """Reset configuration to initial state."""
        self._load_config()