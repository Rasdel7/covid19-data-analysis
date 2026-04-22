import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load Datasets
daily   = pd.read_csv('worldometer_coronavirus_daily_data.csv')
summary = pd.read_csv('worldometer_coronavirus_summary_data.csv')

print("Daily dataset shape   :", daily.shape)
print("Summary dataset shape :", summary.shape)
print("\nDaily columns   :", daily.columns.tolist())
print("Summary columns :", summary.columns.tolist())

daily['date'] = pd.to_datetime(daily['date'])

# Top 10 countries by total cases
top_cases = summary.nlargest(10, 'total_confirmed')

plt.figure(figsize=(12, 6))
colors = sns.color_palette('Reds_r', len(top_cases))
bars = plt.bar(top_cases['country'], top_cases['total_confirmed'] / 1e6,
               color=colors, edgecolor='black')
for bar, val in zip(bars, top_cases['total_confirmed'] / 1e6):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3,
             f'{val:.1f}M', ha='center', fontsize=9, fontweight='bold')
plt.title('Top 10 Countries by Total COVID-19 Cases', fontsize=14)
plt.xlabel('Country')
plt.ylabel('Total Cases (Millions)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top_cases.png')
print("Top cases chart saved!")

# Top 10 by total deaths
top_deaths = summary.nlargest(10, 'total_deaths')

plt.figure(figsize=(12, 6))
colors = sns.color_palette('Oranges_r', len(top_deaths))
bars = plt.barh(top_deaths['country'], top_deaths['total_deaths'] / 1e3,
                color=colors, edgecolor='black')
for bar, val in zip(bars, top_deaths['total_deaths'] / 1e3):
    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val:.0f}K', va='center', fontsize=9, fontweight='bold')
plt.title('Top 10 Countries by Total COVID-19 Deaths', fontsize=14)
plt.xlabel('Total Deaths (Thousands)')
plt.tight_layout()
plt.savefig('top_deaths.png')
print("Top deaths chart saved!")

# Death rate
summary['death_rate'] = (summary['total_deaths'] / summary['total_confirmed']) * 100
top_dr = summary[summary['total_confirmed'] > 100000].nlargest(10, 'death_rate')

plt.figure(figsize=(11, 6))
colors = sns.color_palette('Purples_r', len(top_dr))
bars = plt.bar(top_dr['country'], top_dr['death_rate'],
               color=colors, edgecolor='black')
for bar, val in zip(bars, top_dr['death_rate']):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.05,
             f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
plt.title('Top 10 Countries by COVID-19 Death Rate', fontsize=14)
plt.xlabel('Country')
plt.ylabel('Death Rate (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('death_rate.png')
print("Death rate chart saved!")

# Cases over time — India vs USA vs Brazil vs UK
countries  = ['India', 'USA', 'Brazil', 'UK']
colors_map = {
    'India':  '#FF9933',
    'USA':    '#3C3B6E',
    'Brazil': '#009C3B',
    'UK':     '#012169'
}

plt.figure(figsize=(13, 6))
for country in countries:
    cdf = daily[daily['country'] == country].dropna(
        subset=['cumulative_total_cases'])
    if cdf.empty:
        print(f"Note: {country} not found in daily data, skipping.")
        continue
    plt.plot(cdf['date'], cdf['cumulative_total_cases'] / 1e6,
             label=country, color=colors_map[country], linewidth=2)
plt.title('COVID-19 Total Cases Over Time', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Total Cases (Millions)')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('cases_over_time.png')
print("Cases over time chart saved!")

# India daily new cases with 7-day average
india = daily[daily['country'] == 'India'].dropna(
    subset=['daily_new_cases']).copy()
india['7day_avg'] = india['daily_new_cases'].rolling(7).mean()

plt.figure(figsize=(13, 5))
plt.fill_between(india['date'], india['daily_new_cases'],
                 alpha=0.3, color='#FF9933')
plt.plot(india['date'], india['7day_avg'],
         color='#FF5733', linewidth=2, label='7-Day Average')
plt.title('India Daily COVID-19 Cases with 7-Day Average', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Daily New Cases')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('india_daily_cases.png')
print("India daily cases chart saved!")

# Cases by continent
continent = summary.groupby('continent')['total_confirmed'].sum().sort_values(
    ascending=False)

plt.figure(figsize=(10, 6))
colors = sns.color_palette('Set2', len(continent))
plt.pie(continent.values, labels=continent.index,
        autopct='%1.1f%%', colors=colors,
        startangle=90, wedgeprops={'edgecolor': 'black'})
plt.title('COVID-19 Cases Distribution by Continent', fontsize=14)
plt.tight_layout()
plt.savefig('cases_by_continent.png')
print("Continent chart saved!")

# Key insights
india_s = summary[summary['country'] == 'India'].iloc[0]
top     = summary.nlargest(1, 'total_confirmed').iloc[0]

print("\nKey Insights:")
print(f"  Most cases country : {top['country']} ({top['total_confirmed']/1e6:.1f}M)")
print(f"  India total cases  : {india_s['total_confirmed']/1e6:.2f}M")
print(f"  India death rate   : {india_s['death_rate']:.2f}%")
print(f"  Countries analyzed : {summary['country'].nunique()}")
print("\nEDA Complete! 6 charts saved.")