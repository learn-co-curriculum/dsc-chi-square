# Chi-Square Tests

## Introduction

In this lesson you'll learn about another type of hypothesis test: the chi-square test! Also written as $\chi^2$ *test* or *chi-squared test*, this test is used for making claims about the frequencies of *categorical* data. Because it is testing frequencies rather than population parameters, this test is known as a *non-parametric* test.

## Objectives

You will be able to:

* Identify use cases for the chi-square test
* Distinguish between chi-square tests for goodness of fit, independence, and homogeneity
* Perform a chi-square test and make conclusions about an experiment based on the results


```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
```

## Comparing t-tests and Chi-Square Tests

We'll introduce the chi-square test by comparing to and contrasting it with a familiar statistical test: the t-test!

### t-test Refresher

The t-test is applicable for **continuous** variables that can be represented by a **probability density function** (PDF), which allows us to understand the data in terms of **parameters** such as the mean and variance. There are several different kinds of t-tests depending on the question being asked, although we have mainly focused on one-sample and two-sample (independent) tests.

For example, we might do a one-sample, two-tailed t-test if we have the following (made up) sample of data, and we are trying to determine whether the mean of this data is significantly different from a $\mu_{0}$ of 21.


```python
sample_data = stats.norm.rvs(20, scale=2, size=50, random_state=5)
fig, ax = plt.subplots()

ax.hist(sample_data, alpha=0.5, label="Sample Data")
ax.axvline(sample_data.mean(), linestyle="--", label=r"Sample Mean $\mu$")
ax.axvline(21, linestyle="--", color="orange", label=r"Theoretical Mean $\mu_{0}$")

ax.set_ylabel("Count")

ax.legend();
```


    
![png](index_files/index_5_0.png)
    


Recall that the t-test:

* Has **null and alternative hypotheses about the mean(s)** of one or two samples. For the example shown above, the alternative hypothesis is $\mu \neq \mu_{0}$ (i.e. $\mu \neq 21$)
* Involves the calculation of a **t-statistic** (or t-value) that represents a standardized version of the difference between the two means, utilizing the sample variance as well as the number in the sample to perform this standardization
* Compares this t-statistic to the **t-distribution** (a bell-curve-shaped distribution) in order to determine whether we can reject the null hypothesis at a given alpha level — i.e. to determine whether the difference specified by the alternative hypothesis is statistically significant

The simplest way to execute a t-test is like this, using `scipy.stats`. We pass in the data plotted above, and use the hypotheses $H_{0}: \mu = \mu_{0}$ and $H_{a}: \mu \neq \mu_{0}$ as well as $\alpha = 0.01$ to come to a conclusion.


```python
stats.ttest_1samp(sample_data, 21)
```




    Ttest_1sampResult(statistic=-3.335711380689097, pvalue=0.001628519936938842)



Based on the results above (two-sided, since our alternative hypothesis is $\mu_ \neq \mu_{0}$) we can reject the null hypothesis at an alpha of 0.01, since the resulting p-value (0.0016) is less than our alpha. Therefore we can say that the difference between the sample mean and 21 is statistically significant at the 0.01 significance level.

(If you look closely at the code generating the sample, it was generated using different a df of 20 and with fairly little variance, so it it makes sense that we got this result!)

### Chi-Square Test Introduction

The chi-square ($\chi^2$) test is applicable for **discrete** variables that can be represented by a **probability mass function**, which allows us to understand the data in terms of the **frequencies** of each outcome. There are several different kinds of chi-square tests depending on the question being asked, but we'll focus on *Pearson's chi-square test* and how it is applied for goodness of fit, independence, and homogeneity.

Let's start with a goodness of fit example. This is kind of like the one-sample t-test shown above, in that we are comparing sample data to a theoretical value. This time instead of comparing the sample mean to a theoretical mean, we will compare the frequencies of observed data to the expected frequencies.

For our particular example, let's use the coin toss at the Super Bowl (current data through Super Bowl 55). We expect that this is a "fair" coin, meaning that we would expect it to produce Heads and Tails equally often.


```python
# Data from sportsbettingdime.com
sb_data = pd.read_csv("superbowl.csv")
sb_data.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Super Bowl</th>
      <th>Coin Toss Outcome</th>
      <th>Coin Toss Winner</th>
      <th>Game Winner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>50</th>
      <td>51</td>
      <td>Tails</td>
      <td>Home Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>51</th>
      <td>52</td>
      <td>Heads</td>
      <td>Home Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>52</th>
      <td>53</td>
      <td>Tails</td>
      <td>Home Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>53</th>
      <td>54</td>
      <td>Tails</td>
      <td>Away Team</td>
      <td>Home Team</td>
    </tr>
    <tr>
      <th>54</th>
      <td>55</td>
      <td>Heads</td>
      <td>Away Team</td>
      <td>Home Team</td>
    </tr>
  </tbody>
</table>
</div>




```python
coin_toss_counts = sb_data["Coin Toss Outcome"].value_counts().sort_index()
coin_toss_counts
```




    Heads    26
    Tails    29
    Name: Coin Toss Outcome, dtype: int64




```python
fig, ax = plt.subplots()

# Extract observed counts
observed = coin_toss_counts.values
# Heads and tails each expected half the time
expected = [sum(coin_toss_counts)/2, sum(coin_toss_counts)/2]


# Placeholder data for display purposes; you can ignore thes values
x = np.array([0, 5])
offset = 1
bar_width = 2

# Plot bars
ax.bar(x-offset, observed, bar_width, label="Observed")
ax.bar(x+offset, expected, bar_width, label="Expected")

# Customize appearance
ax.set_xticks(x)
ax.set_xticklabels(["Heads", "Tails"])
ax.set_ylabel("Count")
ax.legend(loc="right");
```


    
![png](index_files/index_12_0.png)
    


As with the previous graph, the orange shows a theoretical (expected) value and the blue shows what we actually observed from our sample.

Unlike the previous graph, the expected value is not a single line representing a parameter, it's a pair of orange bars showing how many times we would expect to see Heads and Tails outcomes.

As you can see, the coin toss at the Super Bowl has had slightly fewer Heads results and slightly more Tails results than we expected. But is that difference statistically significant?

To answer this, we'll need to perform a chi-square test.

A chi-square test:

* Has **null and alternative hypotheses about the frequencies of categorical data**. For the example shown above, we'll use the null hypothesis that $P(Heads) = P(Tails) = 0.5$, i.e. that there is no significant difference between the observed and expected values. The alternative hypothesis is that there is a significant difference.
* Involves the calculation of a **chi-square statistic** (also just referred to as $\chi^2$) that represents a standardized version of the difference between the observed and expected values
* Compares this $\chi^2$ to the **chi-square distribution** (the shape of which varies depending on the degrees of freedom) in order to determine whether we can reject the null hypothesis at a given alpha level — i.e. to determine whether the difference specified by the alternative hypothesis is statistically significant

Once again, the simplest way to do this is using `scipy.stats`. We'll again say that our alpha is 0.01.


```python
result = stats.chisquare(observed, expected)
result
```




    Power_divergenceResult(statistic=0.16363636363636364, pvalue=0.6858304344516056)



Based on the results above, we fail to reject the null hypothesis at our desired significance level. This is because we found a p-value of 0.69, which is higher than our specified $\alpha = 0.01$. (Chi-square tests are always one-tailed so we don't have to consider whether to divide the p-value by 2.)

In other words, we do not have statistically significant evidence that the coin used here is not a "fair" coin! This was an example of a "goodness of fit" application of chi-square.

## Chi-Square Test Calculations

In some cases, you will have extremely limited time for applying statistical tests, and will only have the bandwidth to learn how the null and alternative hypotheses are set up, and the code to execute the test. You'll need to be comfortable with some level of ambiguity, now that you're familiar with statistical tests in general!

But for now, let's break down the previous problem to understand what this statistical test is doing. How did it produce that p-value?

### Chi-Square Statistic

Here is the formula for $\chi^2$:

# $$ \chi^2=\sum_{i=1}^{n} \frac{(O_i - E_i)^2}{E_i} $$

Spelling this out, it means that we are finding the sum of the squared difference between the observed and expected values $(O_i - E_i)^2$ divided by the expected values $E_i$ for all categories $i$. Just like a t-statistic, it is representing the difference between values in a lower-dimensional way.

We'll apply this to our coin flip example below.


```python
n = 2 # number of categories (Heads, Tails)
chi_square = sum([((observed[i] - expected[i])**2)/expected[i] for i in range(n)])
chi_square
```




    0.16363636363636364



Note that this is the same as the statistic from when we called the `chisquare` function:


```python
result.statistic
```




    0.16363636363636364



### Chi-Square Distribution

Below we plot the relevant $\chi^2$ distribution for our current number of categories (degrees of freedom).


```python
# Degrees of freedom
df = n - 1

fig, ax = plt.subplots()
x = np.linspace(0.1, 15, 200) # placeholder x values
y = stats.chi2.pdf(x, df)

ax.plot(x, y, color='darkblue', label=r"$\chi^2$ distribution PDF")
ax.legend();
```


    
![png](index_files/index_22_0.png)
    


Then we can find the critical $\chi^2$ value and plot that as well.


```python
alpha = 0.01
critical_value = stats.chi2.ppf(1-alpha, df=df)
critical_value
```




    6.6348966010212145




```python
fig, ax = plt.subplots()

ax.plot(x, y, color='darkblue', label=r"$\chi^2$ distribution PDF")
ax.axvline(critical_value, color='green', linestyle="--", label=r"critical $\chi^2$")

ax.legend();
```


    
![png](index_files/index_25_0.png)
    


We won't actually plot the rejection region here because it is so small; just know that anywhere under the blue PDF line and to the right of the green line is the rejection region.

Now we'll add our observed chi-square to this plot.


```python
fig, ax = plt.subplots()

ax.plot(x, y, color='darkblue', label=r"$\chi^2$ distribution PDF")
ax.axvline(critical_value, color='green', linestyle="--", label=r"critical $\chi^2$")
ax.axvline(chi_square, color='red', label=r"observed $\chi^2$")

ax.legend();
```


    
![png](index_files/index_27_0.png)
    


To reject the null hypothesis, the observed chi-square would need to be to the right of the critical chi-square, but we can see that it is well to the left.

Alternatively, we could calculate the p-value directly:


```python
stats.chi2.sf(chi_square, df=df)
```




    0.6858304344516056



This is the same value we got from the `chisquare` function:


```python
result.pvalue
```




    0.6858304344516056



## Chi-Square Distributions by Degrees of Freedom

Note that the shape of the chi-square distribution is different depending on the degrees of freedom — it isn't always monotonically decreasing like the graph shown above. Distributions for various degrees of freedom are shown below:


```python
fig, ax = plt.subplots(figsize=(10,10))

for df_ in range(1, 5):
    y_ = stats.chi2.pdf(x, df_)
    ax.plot(x, y_, label=f"df = {df_}")
    
ax.set_xlabel(r'$\chi^2$', fontsize="x-large")
ax.set_ylabel("Probability", fontsize="x-large")
ax.set_title("Chi-Square PDFs for Various Degrees of Freedom")
    
ax.legend(fontsize="x-large");
```


    
![png](index_files/index_33_0.png)
    


## Other Use Cases for Chi-Square Tests

The previous example was similar to a one-sample t-test because we were comparing one set of values to theoretical proportions represented by this concept of a "fair" coin.

The two other ways you can apply chi-square tests are more like two-sample t-tests, where we are comparing two separate samples to understand whether they are different. These techniques are called the test for independence and the test for homogeneity.

### Chi-Square Test for Independence

The chi-square test for independence is an important tool for scientific experimental design, where both the hypothesized independent and dependent variables are categorical rather than numeric.

Returning to our Super Bowl data, let's hypothesize a causal connection between two variables. We hypothesize that winning the coin toss is an independent variable, and winning the game is a dependent variable. In other words, that winning the coin toss and winning the game are not independent variables.

The null hypothesis is the opposite of this, that winning the coin toss and winning the game are independent variables and neither is dependent on the other.

One other way to phrase this as a research question is: **is winning the game related to winning the coin toss, or are these unrelated (independent) variables?**

Let's begin by aggregating the data — conveniently pandas has a `crosstab` function that will set this up for us automatically:


```python
independence_table = pd.crosstab(sb_data["Coin Toss Winner"], sb_data["Game Winner"])
independence_table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Game Winner</th>
      <th>Away Team</th>
      <th>Home Team</th>
    </tr>
    <tr>
      <th>Coin Toss Winner</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Away Team</th>
      <td>15</td>
      <td>15</td>
    </tr>
    <tr>
      <th>Home Team</th>
      <td>16</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



We observe that there is a slight imbalance here, where the Home and Away team seem to win equally often when the Away team was the coin toss winner, but the Home team wins the game less often when the Home team was the coin toss winner.

(This aligns with a superstition/fan theory that winning the coin toss causes a game loss. As of this writing, the coin toss winner had not won the game for the past 7 Super Bowls!)

But, is the difference statistically significant?

Because we have the data set up as a [contingency table](https://en.wikipedia.org/wiki/Contingency_table), we can use the `chi2_contingency` function instead of `chisquare`:


```python
chi2, p, dof, ex = stats.chi2_contingency(independence_table)

print("Chi-square statistic:", chi2)
print("p-value:", p)
```

    Chi-square statistic: 0.5920138888888885
    p-value: 0.44164141533080714


The results are a little different, but we have our familiar chi-square statistic and p-value.

(The other two values returned are `dof`, the degree of freedom and `ex`, the expected frequencies calculated along the way.)

Again assuming $\alpha = 0.01$, we once again fail to reject the null hypothesis (that the variables are independent) because $p \nless \alpha$. We did not find statistically significant evidence that winning the coin toss is the independent variable and winning the game is the dependent variable.

### Chi-Square Test for Homogeneity

One other way we can use a chi-square test is a test for homogeneity. It is very subtly different from the previous test of independence in terms of the framing, but the code is very similar.

Whereas the independence test is about the *factors* (e.g. winning the coin toss vs. winning the game), homogeneity is about the *labels* (values) themselves. Similar to two-sample t-tests, the goal is comparing the distributions of two population samples, to understand whether their underlying populations follow the same distribution.

Let's treat "Home Teams" as one sample and "Away Teams" as another sample. Is the distribution of wins significantly different between these populations?


```python
new_table = []
for index, row in sb_data.iterrows():
    new_row = {}
    if row["Game Winner"] == "Home Team":
        # If the home team won the game, the game loser (away team) called the coin toss
        new_row["Who Called It"] = "Game Loser"
    else:
        # The away team won, so the game winner called the coin toss
        new_row["Who Called It"] = "Game Winner"

    if row["Coin Toss Winner"] == "Home Team":
        # If the home team won the coin toss, the call is the opposite of the outcome
        if row["Coin Toss Outcome"] == "Heads":
            new_row["Coin Toss Call"] = "Tails"
        else:
            new_row["Coin Toss Call"] = "Heads"
    else:
        # If the away team won the coin toss, the call is the same as the outcome
        new_row["Coin Toss Call"] = row["Coin Toss Outcome"]

    new_table.append(new_row)
```


```python
pd.DataFrame(new_table)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Who Called It</th>
      <th>Coin Toss Call</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Game Loser</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Game Winner</td>
      <td>Heads</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Game Loser</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Game Loser</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Game Winner</td>
      <td>Heads</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Game Winner</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Game Winner</td>
      <td>Heads</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Game Loser</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Game Winner</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Game Loser</td>
      <td>Heads</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Game Winner</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Game Winner</td>
      <td>Heads</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Game Winner</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Game Loser</td>
      <td>Heads</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Game Winner</td>
      <td>Heads</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Game Winner</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Game Loser</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Game Loser</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Game Loser</td>
      <td>Heads</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Game Winner</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Game Loser</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Game Winner</td>
      <td>Heads</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Game Loser</td>
      <td>Heads</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Game Winner</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Game Loser</td>
      <td>Heads</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Game Winner</td>
      <td>Heads</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Game Loser</td>
      <td>Heads</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Game Winner</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Game Loser</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Game Winner</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Game Loser</td>
      <td>Heads</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Game Loser</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Game Winner</td>
      <td>Heads</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Game Winner</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Game Winner</td>
      <td>Heads</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Game Loser</td>
      <td>Heads</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Game Loser</td>
      <td>Heads</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Game Loser</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Game Winner</td>
      <td>Heads</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Game Loser</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Game Winner</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Game Winner</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Game Winner</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Game Winner</td>
      <td>Heads</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Game Loser</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Game Winner</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Game Winner</td>
      <td>Heads</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Game Winner</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Game Winner</td>
      <td>Heads</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Game Loser</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Game Winner</td>
      <td>Heads</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Game Winner</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Game Winner</td>
      <td>Heads</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Game Loser</td>
      <td>Tails</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Game Loser</td>
      <td>Heads</td>
    </tr>
  </tbody>
</table>
</div>




```python
sb_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Super Bowl</th>
      <th>Coin Toss Outcome</th>
      <th>Coin Toss Winner</th>
      <th>Game Winner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Heads</td>
      <td>Home Team</td>
      <td>Home Team</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Tails</td>
      <td>Home Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Heads</td>
      <td>Home Team</td>
      <td>Home Team</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Tails</td>
      <td>Away Team</td>
      <td>Home Team</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Tails</td>
      <td>Home Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Heads</td>
      <td>Home Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Heads</td>
      <td>Away Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Heads</td>
      <td>Home Team</td>
      <td>Home Team</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Tails</td>
      <td>Away Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>Heads</td>
      <td>Away Team</td>
      <td>Home Team</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>Tails</td>
      <td>Away Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>Heads</td>
      <td>Away Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>Heads</td>
      <td>Home Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>Heads</td>
      <td>Away Team</td>
      <td>Home Team</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>Tails</td>
      <td>Home Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>Tails</td>
      <td>Away Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>Tails</td>
      <td>Away Team</td>
      <td>Home Team</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>Heads</td>
      <td>Home Team</td>
      <td>Home Team</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>Tails</td>
      <td>Home Team</td>
      <td>Home Team</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>Tails</td>
      <td>Away Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>Tails</td>
      <td>Away Team</td>
      <td>Home Team</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>Heads</td>
      <td>Away Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23</td>
      <td>Tails</td>
      <td>Home Team</td>
      <td>Home Team</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>Heads</td>
      <td>Home Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25</td>
      <td>Heads</td>
      <td>Away Team</td>
      <td>Home Team</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26</td>
      <td>Heads</td>
      <td>Away Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27</td>
      <td>Heads</td>
      <td>Away Team</td>
      <td>Home Team</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>Tails</td>
      <td>Away Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29</td>
      <td>Heads</td>
      <td>Home Team</td>
      <td>Home Team</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>Tails</td>
      <td>Away Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>30</th>
      <td>31</td>
      <td>Heads</td>
      <td>Away Team</td>
      <td>Home Team</td>
    </tr>
    <tr>
      <th>31</th>
      <td>32</td>
      <td>Tails</td>
      <td>Away Team</td>
      <td>Home Team</td>
    </tr>
    <tr>
      <th>32</th>
      <td>33</td>
      <td>Tails</td>
      <td>Home Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>33</th>
      <td>34</td>
      <td>Tails</td>
      <td>Away Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>34</th>
      <td>35</td>
      <td>Tails</td>
      <td>Home Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>35</th>
      <td>36</td>
      <td>Heads</td>
      <td>Away Team</td>
      <td>Home Team</td>
    </tr>
    <tr>
      <th>36</th>
      <td>37</td>
      <td>Tails</td>
      <td>Home Team</td>
      <td>Home Team</td>
    </tr>
    <tr>
      <th>37</th>
      <td>38</td>
      <td>Tails</td>
      <td>Away Team</td>
      <td>Home Team</td>
    </tr>
    <tr>
      <th>38</th>
      <td>39</td>
      <td>Tails</td>
      <td>Home Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>39</th>
      <td>40</td>
      <td>Tails</td>
      <td>Away Team</td>
      <td>Home Team</td>
    </tr>
    <tr>
      <th>40</th>
      <td>41</td>
      <td>Heads</td>
      <td>Home Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>41</th>
      <td>42</td>
      <td>Tails</td>
      <td>Away Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>42</th>
      <td>43</td>
      <td>Heads</td>
      <td>Home Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>43</th>
      <td>44</td>
      <td>Heads</td>
      <td>Away Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>44</th>
      <td>45</td>
      <td>Heads</td>
      <td>Home Team</td>
      <td>Home Team</td>
    </tr>
    <tr>
      <th>45</th>
      <td>46</td>
      <td>Heads</td>
      <td>Home Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>46</th>
      <td>47</td>
      <td>Heads</td>
      <td>Away Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>47</th>
      <td>48</td>
      <td>Tails</td>
      <td>Away Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>48</th>
      <td>49</td>
      <td>Tails</td>
      <td>Home Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>49</th>
      <td>50</td>
      <td>Tails</td>
      <td>Away Team</td>
      <td>Home Team</td>
    </tr>
    <tr>
      <th>50</th>
      <td>51</td>
      <td>Tails</td>
      <td>Home Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>51</th>
      <td>52</td>
      <td>Heads</td>
      <td>Home Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>52</th>
      <td>53</td>
      <td>Tails</td>
      <td>Home Team</td>
      <td>Away Team</td>
    </tr>
    <tr>
      <th>53</th>
      <td>54</td>
      <td>Tails</td>
      <td>Away Team</td>
      <td>Home Team</td>
    </tr>
    <tr>
      <th>54</th>
      <td>55</td>
      <td>Heads</td>
      <td>Away Team</td>
      <td>Home Team</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
