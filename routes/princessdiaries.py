from __future__ import annotations
from typing import List, Tuple
from flask import Flask, request, jsonify
import numpy as np
import logging

from flask import request
from routes import app

@app.route('/princess-diaries', methods=['POST'])
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



"""
{'tasks': [{'name': 'Public Speaking XII', 'start': 1833, 'end': 5491, 'station': 18, 'score': 1}, {'name': 'Nails IX', 'start': 6383, 'end': 6459, 'station': 0, 'score': 5}, {'name': 'Public Speaking XXIV', 'start': 6261, 'end': 6941, 'station': 23, 'score': 6}, {'name': 'Public Speaking XXII', 'start': 4235, 'end': 5643, 'station': 27, 'score': 3}, {'name': 'Public Speaking IX', 'start': 2227, 'end': 2659, 'station': 22, 'score': 3}, {'name': 'Public Speaking VIII', 'start': 67, 'end': 5729, 'station': 23, 'score': 8}, {'name': 'Public Speaking XXX', 'start': 5757, 'end': 6530, 'station': 9, 'score': 10}, {'name': 'Public Speaking XIII', 'start': 4919, 'end': 6539, 'station': 15, 'score': 1}, {'name': 'Public Speaking XXVIV', 'start': 3002, 'end': 6331, 'station': 1, 'score': 7}, {'name': 'Public Speaking II', 'start': 3173, 'end': 5833, 'station': 1, 'score': 8}, {'name': 'Public Speaking XI', 'start': 1855, 'end': 6662, 'station': 7, 'score': 9}, {'name': 'Nails XIX', 'start': 5410, 'end': 6185, 'station': 12, 'score': 6}, {'name': 'Public Speaking XX', 'start': 5700, 'end': 5970, 'station': 6, 'score': 4}, {'name': 'Public Speaking XXVIII', 'start': 2125, 'end': 2577, 'station': 23, 'score': 8}, {'name': 'Public Speaking XIX', 'start': 4025, 'end': 6284, 'station': 9, 'score': 1}, {'name': 'Public Speaking III', 'start': 5344, 'end': 6444, 'station': 17, 'score': 2}, {'name': 'Public Speaking VI', 'start': 2683, 'end': 5815, 'station': 25, 'score': 8}, {'name': 'Public Speaking XXI', 'start': 6648, 'end': 6869, 'station': 9, 'score': 6}, {'name': 'Nails X', 'start': 1473, 'end': 5242, 'station': 18, 'score': 5}, {'name': 'Public Speaking XXIII', 'start': 6242, 'end': 6643, 'station': 13, 'score': 1}, {'name': 'Public Speaking VII', 'start': 6857, 'end': 7058, 'station': 22, 'score': 2}, {'name': 'Public Speaking I', 'start': 3806, 'end': 4025, 'station': 14, 'score': 1}, {'name': 'Public Speaking X', 'start': 2529, 'end': 3515, 'station': 3, 'score': 2}, {'name': 'Public Speaking V', 'start': 980, 'end': 4473, 'station': 14, 'score': 5}, {'name': 'Public Speaking XXVII', 'start': 1715, 'end': 5346, 'station': 27, 'score': 6}, {'name': 'Public Speaking XXV', 'start': 7139, 'end': 7159, 'station': 24, 'score': 2}, {'name': 'Public Speaking XIV', 'start': 5990, 'end': 6610, 'station': 27, 'score': 10}, {'name': 'Public Speaking XXVI', 'start': 3261, 'end': 3463, 'station': 4, 'score': 2}, {'name': 'Public Speaking XVII', 'start': 2895, 'end': 5987, 'station': 23, 'score': 5}, {'name': 'Public Speaking XVIII', 'start': 6490, 'end': 6830, 'station': 24, 'score': 2}, {'name': 'Public Speaking IV', 'start': 33, 'end': 6585, 'station': 1, 'score': 4}, {'name': 'Public Speaking XV', 'start': 5423, 'end': 7049, 'station': 12, 'score': 9}, {'name': 'Public Speaking XVI', 'start': 2114, 'end': 4598, 'station': 1, 'score': 6}], 'subway': [{'connection': [0, 1], 'fee': 19}, {'connection': [0, 3], 'fee': 239}, {'connection': [0, 6], 'fee': 30}, {'connection': [0, 9], 'fee': 633}, {'connection': [1, 3], 'fee': 733}, {'connection': [1, 9], 'fee': 223}, {'connection': [1, 13], 'fee': 126}, {'connection': [1, 23], 'fee': 660}, {'connection': [3, 4], 'fee': 929}, {'connection': [3, 7], 'fee': 586}, {'connection': [4, 6], 'fee': 151}, {'connection': [6, 12], 'fee': 417}, {'connection': [6, 14], 'fee': 553}, {'connection': [6, 17], 'fee': 887}, {'connection': [7, 15], 'fee': 523}, {'connection': [7, 25], 'fee': 101}, {'connection': [9, 15], 'fee': 53}, {'connection': [9, 24], 'fee': 363}, {'connection': [12, 27], 'fee': 975}, {'connection': [13, 14], 'fee': 794}, {'connection': [14, 22], 'fee': 200}, {'connection': [15, 17], 'fee': 131}, {'connection': [15, 18], 'fee': 170}, {'connection': [15, 24], 'fee': 618}, {'connection': [17, 24], 'fee': 84}], 'starting_station': 22}

"""

