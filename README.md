# Histogram of Pclass and fare
titanic_data.hist("Pclass")
plt.xlabel("Passenger Class", fontsize = 14)
plt.ylabel("Passenger Count", fontsize = 14)
plt.title("Number of Passengers per Class", fontsize = 20)

![image](https://github.com/user-attachments/assets/990f8341-8763-4afc-9847-71ebf9b54526)

#Pie chart to display survival rate
survival_rate = titanic_data["Survived"].value_counts()
pie_colors = ["green", "red"]
plt.pie(survival_rate, labels = ["survived", "dead"], colors = pie_colors)
plt.title("Titanic Survival Rate", fontsize = 20)

![image](https://github.com/user-attachments/assets/9000aa7b-69db-4d18-ab8e-4438b170a0c3)

#Scatter plot to show Correlation
plt.scatter(pclass, fare, s=40, color = "pink")
plt.plot(pclass, linewidth=3, color = "black", linestyle = ":", label = "Best Fit")
plt.xlabel("Passenger Class", fontsize = 14)
plt.ylabel("Ticket Fare", fontsize = 14)
plt.title("Ticket Fare per Passenger Class", fontsize = 20)
plt.legend()

![image](https://github.com/user-attachments/assets/ceb41d53-7a56-4d0b-bf07-8da9bc33e572)

# Chi test chart on the sex and survived
sex_survived = pd.crosstab(titanic_data["Sex"], titanic_data["Survived"], normalize= "columns")
print(sex_survived)
c, p, dof, expected = scipy.stats.chi2_contingency(sex_survived)
print("The P-value is:", p)


Age vs. Survival Rate: Analyzing the relationship between a passenger's age and their likelihood of survival can provide insights into whether younger or older passengers had a better chance of surviving. This comparison can help validate historical reports that children were prioritized during evacuation.

# This is for the final project code. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Generate salaries for most players (below $1M)
lower_salaries = np.random.exponential(scale=300000, size=1000)  # Scale controls the spread

# Add a few outliers (high earners)
outliers = np.random.uniform(1000000, 5000000, size=10)  # Salaries between $1M and $5M

# Combine the two datasets
salaries = np.concatenate([lower_salaries, outliers])

# Calculate the median salary
median_salary = np.median(salaries)

# Plot the distribution
plt.figure(figsize=(10, 6))
plt.hist(salaries, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(median_salary, color='red', linestyle='dashed', linewidth=1.5, label=f'Median: ${median_salary:,.0f}')
plt.title('Distribution of Player Salaries', fontsize=16)
plt.xlabel('Salary ($)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# Simulating the dataset (replace this with actual data)
np.random.seed(42)
num_players = 200

data = {
    'Salary': np.random.exponential(scale=500000, size=num_players) + np.random.uniform(0, 3000000, size=num_players),
    'HmRun': np.random.poisson(lam=20, size=num_players) + np.random.randint(0, 10, size=num_players),
    'RBI': np.random.poisson(lam=80, size=num_players) + np.random.randint(0, 20, size=num_players),
    'Putouts': np.random.poisson(lam=100, size=num_players) + np.random.randint(0, 30, size=num_players),
    'Assists': np.random.poisson(lam=60, size=num_players) + np.random.randint(0, 20, size=num_players),
}

df = pd.DataFrame(data)

# Compute correlations
correlations = df.corr()

# Extract specific correlation values
salary_corr = correlations.loc['Salary']
correlation_summary = {
    'HmRun': salary_corr['HmRun'],  # Correlation with Salary
    'RBI': salary_corr['RBI'],
    'Putouts': salary_corr['Putouts'],
    'Assists': salary_corr['Assists']
}

# Print findings
print("Correlation of key metrics with Salary:")
for metric, corr in correlation_summary.items():
    print(f"  {metric}: {corr:.2f}")

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlations, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title('Correlation Matrix of Player Metrics', fontsize=16)
plt.show()

# Simulating the dataset (replace with actual data)
np.random.seed(42)
num_players = 200

data = {
    'Salary': np.random.exponential(scale=500000, size=num_players) + np.random.uniform(0, 3000000, size=num_players),
    'CHits': np.random.poisson(lam=1200, size=num_players) + np.random.randint(0, 500, size=num_players),  # Career Hits
    'CRBI': np.random.poisson(lam=800, size=num_players) + np.random.randint(0, 300, size=num_players),   # Career RBI
    'Hits': np.random.poisson(lam=150, size=num_players),  # Single-season Hits
    'RBI': np.random.poisson(lam=80, size=num_players),    # Single-season RBI
}

df = pd.DataFrame(data)

# Calculate mean and median for Career vs. Single-Season
career_totals_mean = df[['CHits', 'CRBI']].mean()
career_totals_median = df[['CHits', 'CRBI']].median()
single_season_mean = df[['Hits', 'RBI']].mean()
single_season_median = df[['Hits', 'RBI']].median()

# Print summary statistics
print("Career Totals (Mean and Median):")
print(career_totals_mean)
print(career_totals_median)

print("\nSingle-Season Performance (Mean and Median):")
print(single_season_mean)
print(single_season_median)

# Visualizations
plt.figure(figsize=(14, 10))

# Salary vs Career Hits
plt.subplot(2, 2, 1)
sns.regplot(x='CHits', y='Salary', data=df, color='blue', scatter_kws={'alpha': 0.6})
plt.title('Salary vs Career Hits (CHits)', fontsize=14)
plt.xlabel('Career Hits (CHits)', fontsize=12)
plt.ylabel('Salary ($)', fontsize=12)

# Salary vs Career RBI
plt.subplot(2, 2, 2)
sns.regplot(x='CRBI', y='Salary', data=df, color='green', scatter_kws={'alpha': 0.6})
plt.title('Salary vs Career RBI (CRBI)', fontsize=14)
plt.xlabel('Career RBI (CRBI)', fontsize=12)
plt.ylabel('Salary ($)', fontsize=12)

# Salary vs Single-Season Hits
plt.subplot(2, 2, 3)
sns.regplot(x='Hits', y='Salary', data=df, color='orange', scatter_kws={'alpha': 0.6})
plt.title('Salary vs Single-Season Hits', fontsize=14)
plt.xlabel('Single-Season Hits', fontsize=12)
plt.ylabel('Salary ($)', fontsize=12)

# Salary vs Single-Season RBI
plt.subplot(2, 2, 4)
sns.regplot(x='RBI', y='Salary', data=df, color='purple', scatter_kws={'alpha': 0.6})
plt.title('Salary vs Single-Season RBI', fontsize=14)
plt.xlabel('Single-Season RBI', fontsize=12)
plt.ylabel('Salary ($)', fontsize=12)

plt.tight_layout()
plt.show()

# Simulate dataset (replace with actual data)
np.random.seed(42)
num_players = 500

data = {
    'League': np.random.choice(['AL', 'NL'], size=num_players),  # American League or National League
    'Division': np.random.choice(['East', 'Central', 'West'], size=num_players),  # Divisions within leagues
    'Avg': np.random.uniform(0.200, 0.330, size=num_players),  # Offensive average (e.g., batting average)
    'HR': np.random.poisson(lam=20, size=num_players),  # Home Runs
    'RBI': np.random.poisson(lam=80, size=num_players),  # Runs Batted In
}

df = pd.DataFrame(data)

# Group by League and calculate mean averages
league_offense = df.groupby('League')[['Avg', 'HR', 'RBI']].mean()

# Group by League and Division
league_division_offense = df.groupby(['League', 'Division'])[['Avg', 'HR', 'RBI']].mean()

# Print findings
print("League-Wide Offensive Averages:")
print(league_offense)

print("\nLeague and Division Offensive Averages:")
print(league_division_offense)

# Visualize League Analysis
plt.figure(figsize=(12, 6))

# League Offensive Averages
plt.subplot(1, 2, 1)
league_offense['Avg'].plot(kind='bar', color=['blue', 'orange'], alpha=0.7, edgecolor='black')
plt.title('League-Wide Offensive Averages', fontsize=14)
plt.ylabel('Batting Average', fontsize=12)
plt.xlabel('League', fontsize=12)
plt.xticks(rotation=0)

# League and Division Offensive Averages
plt.subplot(1, 2, 2)
league_division_offense['Avg'].unstack(level=0).plot(kind='bar', figsize=(10, 6), alpha=0.7, edgecolor='black')
plt.title('League & Division Offensive Averages', fontsize=14)
plt.ylabel('Batting Average', fontsize=12)
plt.xlabel('Division', fontsize=12)
plt.legend(title='League', fontsize=10)

plt.tight_layout()
plt.show()
