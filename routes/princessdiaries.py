from __future__ import annotations
from typing import List, Tuple
from flask import Flask, request, jsonify
import numpy as np
import math

from flask import request
from routes import app

@app.route('/princessdiaries', methods=['POST'])
def princessdiaries():
    try:
        data = request.get_json()
        
        tasks = data['tasks']
        subway = data['subway']
        starting_station = data['starting_station']
        
        # Extract all unique stations
        stations = set([starting_station])
        for task in tasks:
            stations.add(task['station'])
        for route in subway:
            stations.update(route['connection'])
        
        # Initialize distance matrix with infinity
        INF = float('inf')
        dist = {}
        for s1 in stations:
            dist[s1] = {}
            for s2 in stations:
                if s1 == s2:
                    dist[s1][s2] = 0
                else:
                    dist[s1][s2] = INF
        
        # Set direct edge distances
        for edge in subway:
            u, v = edge['connection']
            fee = edge['fee']
            dist[u][v] = fee
            dist[v][u] = fee  # Undirected graph
        
        # Floyd-Warshall algorithm
        for k in stations:
            for i in stations:
                for j in stations:
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        # Find optimal schedule using dynamic programming
        if not tasks:
            return jsonify({
                "max_score": 0,
                "min_fee": 0,
                "schedule": []
            })
        
        # Sort tasks by start time
        sorted_tasks = sorted(tasks, key=lambda x: x['start'])
        n = len(sorted_tasks)
        
        # DP to find maximum score schedules
        # dp[i] = (max_score, min_cost_for_max_score, schedule)
        dp = [(0, 0, [])] * n
        
        for i in range(n):
            current_task = sorted_tasks[i]
            
            # Option 1: Don't take current task
            if i > 0:
                dp[i] = dp[i-1]
            else:
                dp[i] = (0, 0, [])
            
            # Option 2: Take current task
            # Find the latest non-overlapping task
            latest_non_overlap = -1
            for j in range(i-1, -1, -1):
                if sorted_tasks[j]['end'] <= current_task['start']:
                    latest_non_overlap = j
                    break
            
            # Calculate score and cost if we take current task
            if latest_non_overlap == -1:
                # First task or no previous non-overlapping task
                prev_score = 0
                prev_cost = 0
                prev_schedule = []
                prev_station = starting_station
            else:
                prev_score, prev_cost, prev_schedule = dp[latest_non_overlap]
                if prev_schedule:
                    prev_station = next(t['station'] for t in sorted_tasks if t['name'] == prev_schedule[-1])
                else:
                    prev_station = starting_station
            
            new_score = prev_score + current_task['score']
            travel_cost = dist[prev_station][current_task['station']]
            new_cost = prev_cost + travel_cost
            new_schedule = prev_schedule + [current_task['name']]
            
            # Update dp[i] if this option is better
            current_score, current_cost, current_schedule = dp[i]
            
            if (new_score > current_score or 
                (new_score == current_score and new_cost < current_cost)):
                dp[i] = (new_score, new_cost, new_schedule)
        
        max_score, cost_without_return, schedule = dp[-1]
        
        # Add return cost to starting station
        if schedule:
            last_task_name = schedule[-1]
            last_station = next(t['station'] for t in sorted_tasks if t['name'] == last_task_name)
            return_cost = dist[last_station][starting_station]
            total_cost = cost_without_return + return_cost
        else:
            total_cost = 0
        
        return jsonify({
            "max_score": max_score,
            "min_fee": total_cost,
            "schedule": schedule
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500