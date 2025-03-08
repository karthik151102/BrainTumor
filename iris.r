# Given data
mu <- 18.5  # Population mean
sample_mean <- 17.85  # Sample mean
sd <- 1.955  # Standard deviation
n <- 14  # Sample size

# Calculate the t-statistic manually
t_stat <- (sample_mean - mu) / (sd / sqrt(n))

# Output the t-statistic
print(t_stat)

# Calculate degrees of freedom
df <- n - 1  

# Calculate p-value for two-tailed test
p_value <- 2 * pt(-abs(t_stat), df)

# Output the p-value
print(p_value)