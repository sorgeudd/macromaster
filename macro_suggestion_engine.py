"""Contextual Macro Suggestion Engine for intelligent macro recommendations"""
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

class MacroContext:
    def __init__(self):
        self.timestamp: datetime
        self.window_title: str = ""
        self.active_application: str = ""
        self.recent_keys: List[str] = []
        self.frequency: int = 0

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat() if hasattr(self, 'timestamp') else None,
            'window_title': self.window_title,
            'active_application': self.active_application,
            'recent_keys': self.recent_keys,
            'frequency': self.frequency
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'MacroContext':
        context = cls()
        if data.get('timestamp'):
            context.timestamp = datetime.fromisoformat(data['timestamp'])
        context.window_title = data.get('window_title', '')
        context.active_application = data.get('active_application', '')
        context.recent_keys = data.get('recent_keys', [])
        context.frequency = data.get('frequency', 0)
        return context

class MacroSuggestionEngine:
    def __init__(self, macros_dir: Path):
        self.logger = logging.getLogger('MacroSuggestionEngine')
        self.macros_dir = macros_dir
        self.context_data_file = self.macros_dir / 'macro_contexts.json'
        self.macro_contexts: Dict[str, List[MacroContext]] = {}
        self._load_contexts()

    def _load_contexts(self):
        """Load saved macro contexts"""
        try:
            if self.context_data_file.exists():
                with open(self.context_data_file, 'r') as f:
                    data = json.load(f)
                    for macro_name, contexts in data.items():
                        self.macro_contexts[macro_name] = [
                            MacroContext.from_dict(ctx) for ctx in contexts
                        ]
                self.logger.info(f"Loaded {len(self.macro_contexts)} macro contexts")
        except Exception as e:
            self.logger.error(f"Error loading macro contexts: {e}")
            self.macro_contexts = {}

    def _save_contexts(self):
        """Save current macro contexts"""
        try:
            data = {
                macro_name: [ctx.to_dict() for ctx in contexts]
                for macro_name, contexts in self.macro_contexts.items()
            }
            with open(self.context_data_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Saved {len(self.macro_contexts)} macro contexts")
        except Exception as e:
            self.logger.error(f"Error saving macro contexts: {e}")

    def update_context(self, macro_name: str, window_title: str, active_app: str, recent_keys: List[str]):
        """Update context information for a macro"""
        try:
            context = MacroContext()
            context.timestamp = datetime.now()
            context.window_title = window_title
            context.active_application = active_app
            context.recent_keys = recent_keys

            if macro_name not in self.macro_contexts:
                self.macro_contexts[macro_name] = []

            # Add new context and update frequency
            for existing_context in self.macro_contexts[macro_name]:
                if (existing_context.window_title == window_title and 
                    existing_context.active_application == active_app):
                    existing_context.frequency += 1
                    existing_context.timestamp = context.timestamp
                    break
            else:
                context.frequency = 1
                self.macro_contexts[macro_name].append(context)

            self._save_contexts()
            self.logger.info(f"Updated context for macro: {macro_name}")

        except Exception as e:
            self.logger.error(f"Error updating context for macro {macro_name}: {e}")

    def get_suggestions(self, window_title: str, active_app: str, recent_keys: List[str], max_suggestions: int = 3) -> List[str]:
        """Get macro suggestions based on current context"""
        try:
            scores = []
            for macro_name, contexts in self.macro_contexts.items():
                max_score = 0
                for context in contexts:
                    score = 0
                    
                    # Score based on exact window title match
                    if context.window_title == window_title:
                        score += 3
                    # Score based on partial window title match
                    elif window_title in context.window_title or context.window_title in window_title:
                        score += 1

                    # Score based on application match
                    if context.active_application == active_app:
                        score += 2

                    # Score based on recent keys match
                    common_keys = set(context.recent_keys) & set(recent_keys)
                    score += len(common_keys) * 0.5

                    # Factor in frequency
                    score *= (1 + context.frequency * 0.1)

                    max_score = max(max_score, score)
                
                if max_score > 0:
                    scores.append((macro_name, max_score))

            # Sort by score and return top suggestions
            scores.sort(key=lambda x: x[1], reverse=True)
            suggestions = [name for name, _ in scores[:max_suggestions]]
            
            self.logger.info(f"Generated {len(suggestions)} suggestions for context")
            return suggestions

        except Exception as e:
            self.logger.error(f"Error generating suggestions: {e}")
            return []

    def clear_old_contexts(self, days_threshold: int = 30):
        """Clear contexts older than the specified threshold"""
        try:
            threshold = datetime.now()
            for macro_name in list(self.macro_contexts.keys()):
                self.macro_contexts[macro_name] = [
                    ctx for ctx in self.macro_contexts[macro_name]
                    if (threshold - ctx.timestamp).days < days_threshold
                ]
                if not self.macro_contexts[macro_name]:
                    del self.macro_contexts[macro_name]
            
            self._save_contexts()
            self.logger.info("Cleaned up old context data")
        except Exception as e:
            self.logger.error(f"Error clearing old contexts: {e}")
