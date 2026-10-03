# LinkedIn launch post (draft)

I built a marketing mix model and then tried to catch it lying.

Most budget decisions still run on last-click or platform-reported ROI. That reporting can't
see sales that would have happened anyway, so every channel looks better than it is and the
money drifts toward whatever sits closest to checkout.

MixLab is my attempt at the alternative: a Bayesian MMM with a budget optimizer and an AI
layer that explains the results to a CMO. The part I care most about is that it can be checked.
I generated synthetic brands where I set every channel's true effect, then scored the model
against them.

What I found:

→ The true ROI landed inside the model's 94% range for 27 of 30 channel estimates.
→ On 13 weeks it had never seen, prediction error was about 1.4%.
→ Naive same-week attribution overstated Meta's ROI five-fold.
→ The optimizer was right about direction and wrong about size: it promised +7.4% and the
truth delivered +3.3%.
→ With no limits on how far channels could move, it promised +16% and delivered nothing. The
"no channel moves more than 30%" rule turned out to be what makes the advice safe.
→ One simulated 8-week lift test narrowed Google Search's ROI range by 85%.

The lesson I'd pass on: a model tells you where to look and what's at stake; an experiment
settles it. Neither is enough alone.

Stack: PyMC-Marketing, Streamlit, Plotly, Claude for the explanation layer (every number it
writes is checked against the model's output).

Dashboard: https://mixlab.streamlit.app/
Code and case study: https://github.com/adityajamdhade6/mixlab

#MarketingAnalytics #MarketingMixModeling #BayesianStatistics #DataScience
