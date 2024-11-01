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



Age vs. Survival Rate: Analyzing the relationship between a passenger's age and their likelihood of survival can provide insights into whether younger or older passengers had a better chance of surviving. This comparison can help validate historical reports that children were prioritized during evacuation.
