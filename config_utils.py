"""
Configuration utilities for the Universal Column Mapper.
This module provides functions to import, export, and manage field configurations.
"""

import json
import streamlit as st
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ConfigurationManager:
    """Manages field configurations and presets."""
    
    @staticmethod
    def load_sample_configurations() -> Dict:
        """Load sample configurations from JSON file."""
        try:
            with open('sample_configurations.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Sample configurations file not found")
            return {"configurations": {}}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing sample configurations: {e}")
            return {"configurations": {}}
    
    @staticmethod
    def validate_configuration(config: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check if config has required structure
        if not isinstance(config, dict):
            errors.append("Configuration must be a dictionary")
            return False, errors
        
        # Check for fields key
        if 'fields' not in config:
            errors.append("Configuration must contain 'fields' key")
            return False, errors
        
        fields = config['fields']
        if not isinstance(fields, dict):
            errors.append("'fields' must be a dictionary")
            return False, errors
        
        # Validate each field
        for field_name, field_description in fields.items():
            if not isinstance(field_name, str) or not field_name.strip():
                errors.append(f"Field name must be non-empty string, got: {field_name}")
            
            if not isinstance(field_description, str) or not field_description.strip():
                errors.append(f"Field description for '{field_name}' must be non-empty string")
        
        # Check if mapping exists and is valid
        if 'mapping' in config:
            mapping = config['mapping']
            if not isinstance(mapping, dict):
                errors.append("'mapping' must be a dictionary")
            else:
                # Validate mapping keys match fields
                for field_name in mapping.keys():
                    if field_name not in fields:
                        errors.append(f"Mapping field '{field_name}' not found in fields definition")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def export_configuration(fields: Dict[str, str], mapping: Optional[Dict[str, str]] = None, 
                           name: str = "Custom Configuration", 
                           description: str = "User-defined field configuration") -> str:
        """
        Export field configuration to JSON format.
        
        Args:
            fields: Dictionary of field names and descriptions
            mapping: Optional mapping of fields to columns
            name: Configuration name
            description: Configuration description
            
        Returns:
            JSON string representation of the configuration
        """
        config = {
            "name": name,
            "description": description,
            "fields": fields
        }
        
        if mapping:
            config["mapping"] = mapping
        
        return json.dumps(config, indent=2, ensure_ascii=False)
    
    @staticmethod
    def import_configuration(json_str: str) -> Tuple[bool, Dict, List[str]]:
        """
        Import configuration from JSON string.
        
        Args:
            json_str: JSON string containing configuration
            
        Returns:
            Tuple of (success, configuration_dict, error_messages)
        """
        try:
            config = json.loads(json_str)
            is_valid, errors = ConfigurationManager.validate_configuration(config)
            
            if is_valid:
                return True, config, []
            else:
                return False, {}, errors
                
        except json.JSONDecodeError as e:
            return False, {}, [f"Invalid JSON format: {str(e)}"]
        except Exception as e:
            return False, {}, [f"Error importing configuration: {str(e)}"]

def add_configuration_ui():
    """Add configuration import/export UI to Streamlit app."""
    
    st.sidebar.header("🔧 Configuration Management")
    
    # Configuration export
    with st.sidebar.expander("📤 Export Configuration"):
        if st.session_state.get('field_descriptions'):
            config_name = st.text_input("Configuration Name", value="My Custom Fields")
            config_desc = st.text_area("Description", value="Custom field configuration")
            
            if st.button("Generate Export", key="export_config"):
                exported_config = ConfigurationManager.export_configuration(
                    st.session_state.field_descriptions,
                    st.session_state.get('mapping_result'),
                    config_name,
                    config_desc
                )
                
                st.download_button(
                    label="📥 Download Configuration",
                    data=exported_config,
                    file_name=f"{config_name.lower().replace(' ', '_')}_config.json",
                    mime="application/json",
                    key="download_config"
                )
        else:
            st.info("Configure fields first to enable export")
    
    # Configuration import
    with st.sidebar.expander("📤 Import Configuration"):
        uploaded_config = st.file_uploader(
            "Upload Configuration File",
            type=['json'],
            key="config_upload",
            help="Upload a previously exported configuration file"
        )
        
        if uploaded_config is not None:
            try:
                config_content = uploaded_config.read().decode('utf-8')
                success, config, errors = ConfigurationManager.import_configuration(config_content)
                
                if success:
                    st.success("✅ Configuration loaded successfully!")
                    
                    # Show configuration preview
                    st.write("**Fields to be loaded:**")
                    for field_name in config['fields'].keys():
                        st.write(f"• {field_name}")
                    
                    if st.button("Apply Configuration", key="apply_config"):
                        st.session_state.field_descriptions = config['fields']
                        st.session_state.required_fields = list(config['fields'].keys())
                        st.session_state.custom_fields = config['fields'].copy()
                        
                        # Apply mapping if available
                        if 'mapping' in config:
                            st.session_state.mapping_result = config['mapping']
                        
                        st.success("Configuration applied successfully!")
                        st.rerun()
                        
                else:
                    st.error("❌ Invalid configuration file")
                    for error in errors:
                        st.error(f"• {error}")
                        
            except Exception as e:
                st.error(f"Error reading configuration file: {e}")
    
    # Sample configurations
    with st.sidebar.expander("📋 Sample Configurations"):
        samples = ConfigurationManager.load_sample_configurations()
        
        if samples.get('configurations'):
            selected_sample = st.selectbox(
                "Choose Sample Configuration",
                ["None"] + list(samples['configurations'].keys()),
                key="sample_config_select"
            )
            
            if selected_sample != "None":
                sample_config = samples['configurations'][selected_sample]
                
                st.write(f"**{sample_config.get('name', selected_sample)}**")
                st.write(sample_config.get('description', 'No description available'))
                
                st.write("**Fields:**")
                for field_name in sample_config['fields'].keys():
                    st.write(f"• {field_name}")
                
                if st.button(f"Load {selected_sample}", key=f"load_sample_{selected_sample}"):
                    st.session_state.field_descriptions = sample_config['fields']
                    st.session_state.required_fields = list(sample_config['fields'].keys())
                    st.session_state.custom_fields = sample_config['fields'].copy()
                    
                    st.success(f"Loaded {sample_config.get('name', selected_sample)} configuration!")
                    st.rerun()
        else:
            st.info("No sample configurations available")

def display_configuration_stats():
    """Display configuration statistics in the main area."""
    if st.session_state.get('field_descriptions'):
        field_count = len(st.session_state.field_descriptions)
        mapped_count = 0
        
        if st.session_state.get('mapping_result'):
            mapped_count = sum(1 for v in st.session_state.mapping_result.values() if v is not None)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Fields", field_count)
        
        with col2:
            st.metric("Mapped Fields", mapped_count)
        
        with col3:
            if field_count > 0:
                completion = (mapped_count / field_count) * 100
                st.metric("Completion", f"{completion:.1f}%")
            else:
                st.metric("Completion", "0%")