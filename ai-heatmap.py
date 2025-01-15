#!/usr/bin/env python
# coding: utf-8

from datetime import datetime, timezone, timedelta
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Optional
import json
import pytz
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# Configuration
CONVO_FOLDER = '/Users/travisf/Documents/projects/gpt-heatmap/chats'
LOCAL_TZ = 'US/Mountain'

@dataclass
class Config:
    convo_folder: str = CONVO_FOLDER
    local_tz: str = LOCAL_TZ

class ConversationData:
    def __init__(self, config: Config):
        self.config = config
        self.conversations = self._load_conversations()
        self.convo_times = self._process_timestamps()
    
    def _load_conversations(self) -> List[Dict]:
        with open(f'{self.config.convo_folder}/conversations.json', 'r') as f:
            return json.load(f)
    
    def _process_timestamps(self) -> List[datetime]:
        return [
            datetime.fromtimestamp(conv['create_time'], tz=timezone.utc)
            .astimezone(pytz.timezone(self.config.local_tz))
            for conv in self.conversations
        ]

class HeatmapVisualizer:
    def __init__(self, convo_times: List[datetime]):
        self.convo_times = convo_times
    
    def create_year_heatmap(self, year: int) -> None:
        dates = [convo.date() for convo in self.convo_times if convo.year == year]
        date_counts = Counter(dates)
        
        if not date_counts:
            print(f"No conversations found for year {year}")
            return
        
        # Create calendar grid
        start_date = datetime(year, 1, 1).date()
        end_date = datetime(year, 12, 31).date()
        date_range = [start_date + timedelta(days=i) 
                     for i in range((end_date - start_date).days + 1)]
        
        # Prepare plot data
        plot_data = [
            (
                ((date - start_date).days + start_date.weekday()) // 7,
                date.weekday(),
                date_counts.get(date, 0)
            )
            for date in date_range
        ]
        
        self._plot_heatmap(plot_data, date_counts, start_date, end_date, year)
    
    def _plot_heatmap(self, plot_data, date_counts, start_date, end_date, year):
        weeks_in_year = (end_date - start_date).days // 7 + 1
        
        # Setup plot
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        ax.set_aspect('equal')
        
        # Calculate metrics for coloring
        max_count_date = max(date_counts, key=date_counts.get)
        max_count = date_counts[max_count_date]
        p90_count = np.percentile(list(date_counts.values()), 90)
        
        # Draw rectangles
        for week, day_of_week, count in plot_data:
            color = plt.cm.Greens((count + 1) / p90_count) if count > 0 else 'lightgray'
            rect = patches.Rectangle(
                (week, day_of_week), 1, 1,
                linewidth=0.5, edgecolor='black', facecolor=color
            )
            ax.add_patch(rect)
        
        # Add month labels
        self._add_month_labels(start_date, end_date, ax)
        
        # Finalize plot
        self._finalize_plot(ax, weeks_in_year, year, date_counts, max_count_date, max_count)
    
    def _add_month_labels(self, start_date, end_date, ax):
        total_days = (end_date - start_date).days + 1
        month_starts = [
            start_date + timedelta(days=i)
            for i in range(total_days)
            if (start_date + timedelta(days=i)).day == 1
        ]
        
        for month_start in month_starts:
            week = (month_start - start_date).days // 7
            plt.text(
                week + 0.5, 7.75,
                month_start.strftime('%b'),
                ha='center', va='center',
                fontsize=10, rotation=0
            )
    
    def _finalize_plot(self, ax, weeks_in_year, year, date_counts, max_count_date, max_count):
        ax.set_xlim(-0.5, weeks_in_year + 0.5)
        ax.set_ylim(-0.5, 8.5)
        plt.title(
            f'{year} ChatGPT Conversation Heatmap '
            f'(total={sum(date_counts.values())}).\n'
            f'Most active day: {max_count_date} with {max_count} convos.',
            fontsize=16
        )
        plt.xticks([])
        plt.yticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        plt.gca().invert_yaxis()
        plt.show()

def main():
    # Configuration
    config = Config()  # Will use the default values defined above
    
    # Load and process data
    data = ConversationData(config)
    
    # Create visualizations
    visualizer = HeatmapVisualizer(data.convo_times)
    for year in [2023, 2024, 2025]:
        visualizer.create_year_heatmap(year)

if __name__ == '__main__':
    main()