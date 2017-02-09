---
title: Gradient Descent and SGD
shorttitle: Gradient Descent and SGD
notebook: gradientdescent.ipynb
noline: 1
layout: wiki
---



```python
%matplotlib inline
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import stats 

from sklearn.datasets.samples_generator import make_regression 
```




A lot of the animations here were adapted from: http://tillbergmann.com/blog/python-gradient-descent.html

A great discussion (and where momentum image was stolen from) is at http://sebastianruder.com/optimizing-gradient-descent/

Gradient descent is one of the most popular algorithms to perform optimization and by far the most common way to optimize neural networks. Gradient descent is a way to minimize an objective function $J_{\theta}$ parameterized by a model's parameters $\theta \in \mathbb{R}^d$ by updating the parameters in the opposite direction of the gradient of the objective function $\nabla_J J(\theta)$ w.r.t. to the parameters. The learning rate $\eta$ determines the size of the steps we take to reach a (local) minimum. In other words, we follow the direction of the slope of the surface created by the objective function downhill until we reach a valley.

There are three variants of gradient descent, which differ in how much data we use to compute the gradient of the objective function. Depending on the amount of data, we make a trade-off between the accuracy of the parameter update and the time it takes to perform an update.

## Example: Linear regression

Let's see briefly how gradient descent can be useful to us in least squares regression. Let's asssume we have an output variable $y$ which we think depends linearly on the input vector $x$. We approximate $y$ by

$$f_\theta (x) =\theta^T x$$

The cost function for our linear least squares regression will then be

$$J(\theta) = \frac{1}{2} \sum_{i=1}^m (f_\theta (x^{(i)}-y^{(i)})^2$$


We create a regression problem using sklearn's `make_regression` function:



```python
#code adapted from http://tillbergmann.com/blog/python-gradient-descent.html
x, y = make_regression(n_samples = 100, 
                       n_features=1, 
                       n_informative=1, 
                       noise=20,
                       random_state=2017)
```




```python
x = x.flatten()
```




```python
slope, intercept, _,_,_ = stats.linregress(x,y)
best_fit = np.vectorize(lambda x: x * slope + intercept)
```




```python
plt.plot(x,y, 'o', alpha=0.5)
grid = np.arange(-3,3,0.1)
plt.plot(grid,best_fit(grid), '.')
```





    [<matplotlib.lines.Line2D at 0x116a1db70>]




![png](gradientdescent_files/gradientdescent_7_1.png)


## Batch gradient descent

Assume that we have a vector of paramters $\theta$ and a cost function $J(\theta)$ which is simply the variable we want to minimize (our objective function). Typically, we will find that the objective function has the form:

$$J(\theta) =\sum_{i=1}^m J_i(\theta)$$

where $J_i$ is associated with the i-th observation in our data set. The batch gradient descent algorithm, starts with some initial feasible  $\theta$ (which we can either fix or assign randomly) and then repeatedly performs the update:

$$\theta := \theta - \eta \nabla_{\theta} J(\theta) = \theta -\eta \sum_{i=1}^m \nabla J_i(\theta)$$

where $\eta$ is a constant controlling step-size and is called the learning rate. Note that in order to make a single update, we need to calculate the gradient using the entire dataset. This can be very inefficient for large datasets.

In code, batch gradient descent looks like this:

```python
for i in range(n_epochs):
  params_grad = evaluate_gradient(loss_function, data, params)
  params = params - learning_rate * params_grad`
```
  
For a given number of epochs $n_{epochs}$, we first evaluate the gradient vector of the loss function using **ALL** examples in the data set, and then we update the parameters with a given learning rate. This is where Theano and automatic differentiation come in handy, and you will learn about them in lab.

Batch gradient descent is guaranteed to converge to the global minimum for convex error surfaces and to a local minimum for non-convex surfaces.

In the linear example it's easy to see that our update step then takes the form:

$$\theta_j := \theta_j + \alpha \sum_{i=1}^m (y^{(i)}-f_\theta (x^{(i)})) x_j^{(i)}$$
for every $j$ (note $\theta_j$ is simply the j-th component of the $\theta$ vector).




```python
def gradient_descent(x, y, theta_init, step=0.001, maxsteps=0, precision=0.001, ):
    costs = []
    m = y.size # number of data points
    theta = theta_init
    history = [] # to store all thetas
    preds = []
    counter = 0
    oldcost = 0
    pred = np.dot(x, theta)
    error = pred - y 
    currentcost = np.sum(error ** 2) / (2 * m)
    preds.append(pred)
    costs.append(currentcost)
    history.append(theta)
    counter+=1
    while abs(currentcost - oldcost) > precision:
        oldcost=currentcost
        gradient = x.T.dot(error)/m 
        theta = theta - step * gradient  # update
        history.append(theta)
        
        pred = np.dot(x, theta)
        error = pred - y 
        currentcost = np.sum(error ** 2) / (2 * m)
        costs.append(currentcost)
        
        if counter % 25 == 0: preds.append(pred)
        counter+=1
        if maxsteps:
            if counter == maxsteps:
                break
        
    return history, costs, preds, counter
```




```python
xaug = np.c_[np.ones(x.shape[0]), x]
theta_i = [-15, 40] + np.random.rand(2)
history, cost, preds, iters = gradient_descent(xaug, y, theta_i, step=0.1)
theta = history[-1]
```




```python
print("Gradient Descent: {:.2f}, {:.2f} {:d}".format(theta[0], theta[1], iters))
print("Least Squares: {:.2f}, {:.2f}".format(intercept, slope))
```


    Gradient Descent: -3.73, 82.80 73
    Least Squares: -3.71, 82.90


One can plot the reduction of cost:



```python
plt.plot(range(len(cost)), cost);
```



![png](gradientdescent_files/gradientdescent_13_0.png)


The following animation shows how the regression line forms:



```python
from JSAnimation import IPython_display
```




```python
def init():
    line.set_data([], [])
    return line,

def animate(i):
    ys = preds[i]
    line.set_data(xaug[:, 1], ys)
    return line,



fig = plt.figure(figsize=(10,6))
ax = plt.axes(xlim=(-3, 2.5), ylim=(-170, 170))
ax.plot(xaug[:,1],y, 'o')
line, = ax.plot([], [], lw=2)
plt.plot(xaug[:,1], best_fit(xaug[:,1]), 'k-', color = "r")

anim = animation.FuncAnimation(fig, animate, init_func=init,
                        frames=len(preds), interval=100)
anim.save('images/gdline.mp4')
anim
```






<script language="javascript">
  /* Define the Animation class */
  function Animation(frames, img_id, slider_id, interval, loop_select_id){
    this.img_id = img_id;
    this.slider_id = slider_id;
    this.loop_select_id = loop_select_id;
    this.interval = interval;
    this.current_frame = 0;
    this.direction = 0;
    this.timer = null;
    this.frames = new Array(frames.length);

    for (var i=0; i<frames.length; i++)
    {
     this.frames[i] = new Image();
     this.frames[i].src = frames[i];
    }
    document.getElementById(this.slider_id).max = this.frames.length - 1;
    this.set_frame(this.current_frame);
  }

  Animation.prototype.get_loop_state = function(){
    var button_group = document[this.loop_select_id].state;
    for (var i = 0; i < button_group.length; i++) {
        var button = button_group[i];
        if (button.checked) {
            return button.value;
        }
    }
    return undefined;
  }

  Animation.prototype.set_frame = function(frame){
    this.current_frame = frame;
    document.getElementById(this.img_id).src = this.frames[this.current_frame].src;
    document.getElementById(this.slider_id).value = this.current_frame;
  }

  Animation.prototype.next_frame = function()
  {
    this.set_frame(Math.min(this.frames.length - 1, this.current_frame + 1));
  }

  Animation.prototype.previous_frame = function()
  {
    this.set_frame(Math.max(0, this.current_frame - 1));
  }

  Animation.prototype.first_frame = function()
  {
    this.set_frame(0);
  }

  Animation.prototype.last_frame = function()
  {
    this.set_frame(this.frames.length - 1);
  }

  Animation.prototype.slower = function()
  {
    this.interval /= 0.7;
    if(this.direction > 0){this.play_animation();}
    else if(this.direction < 0){this.reverse_animation();}
  }

  Animation.prototype.faster = function()
  {
    this.interval *= 0.7;
    if(this.direction > 0){this.play_animation();}
    else if(this.direction < 0){this.reverse_animation();}
  }

  Animation.prototype.anim_step_forward = function()
  {
    this.current_frame += 1;
    if(this.current_frame < this.frames.length){
      this.set_frame(this.current_frame);
    }else{
      var loop_state = this.get_loop_state();
      if(loop_state == "loop"){
        this.first_frame();
      }else if(loop_state == "reflect"){
        this.last_frame();
        this.reverse_animation();
      }else{
        this.pause_animation();
        this.last_frame();
      }
    }
  }

  Animation.prototype.anim_step_reverse = function()
  {
    this.current_frame -= 1;
    if(this.current_frame >= 0){
      this.set_frame(this.current_frame);
    }else{
      var loop_state = this.get_loop_state();
      if(loop_state == "loop"){
        this.last_frame();
      }else if(loop_state == "reflect"){
        this.first_frame();
        this.play_animation();
      }else{
        this.pause_animation();
        this.first_frame();
      }
    }
  }

  Animation.prototype.pause_animation = function()
  {
    this.direction = 0;
    if (this.timer){
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  Animation.prototype.play_animation = function()
  {
    this.pause_animation();
    this.direction = 1;
    var t = this;
    if (!this.timer) this.timer = setInterval(function(){t.anim_step_forward();}, this.interval);
  }

  Animation.prototype.reverse_animation = function()
  {
    this.pause_animation();
    this.direction = -1;
    var t = this;
    if (!this.timer) this.timer = setInterval(function(){t.anim_step_reverse();}, this.interval);
  }
</script>

<div class="animation" align="center">
    <img id="_anim_imgGXFDZGNIPCQHGVWJ">
    <br>
    <input id="_anim_sliderGXFDZGNIPCQHGVWJ" type="range" style="width:350px" name="points" min="0" max="1" step="1" value="0" onchange="animGXFDZGNIPCQHGVWJ.set_frame(parseInt(this.value));"></input>
    <br>
    <button onclick="animGXFDZGNIPCQHGVWJ.slower()">&#8211;</button>
    <button onclick="animGXFDZGNIPCQHGVWJ.first_frame()"><img class="anim_icon" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAQAAAAngNWGAAAAAXNSR0IArs4c6QAAAAJiS0dEAP+Hj8y/AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3QURCAgaeZk4EQAAASlJREFUKM/dkj9LQnEUhp9zr3bpj1uBcKGiJWxzLWivKAIRjIhcCqcgqJbKRagPICiVSVEuNTu0tLYGUg4tkRGUdxLJ0u79Ndxr5FfwTO/L+xzO4XCgO+v2T70AFU+/A/Dhmlzg6Pr0DKAMwOH4zQxAAbAkv2xNeF2RoQUVc1ytgttXUbWVdN1dOPE8pz4j4APQsdFtKA0WY6vpKjqvVciHnvZTS6Ja4HgggJLs7MHxl9nCh8NYcO+iGG0agiaC4h9oa6Vsw2yiK+QHSZT934YoEQABNBcTNDszsrhm1m1B+bFS86PT6QFppx6oeSaeOwlMXRp1h4aK13Y2kuHhUo9ykPboPvFjeEvsrhTMt3ylHyB0r8KZyYdCrbfj4OveoHMANjuyx+76rV+/blxKMZUnLgAAAABJRU5ErkJggg=="></button>
    <button onclick="animGXFDZGNIPCQHGVWJ.previous_frame()"><img class="anim_icon" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAQAAAAngNWGAAAAAXNSR0IArs4c6QAAAAJiS0dEAP+Hj8y/AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3QURCAgyTCyQ6wAAANRJREFUKM9jYBjO4AiUfgzFGGAp4+yayUvX6jMwMDCsYmBgOCS4OAOrSYmMgcc8/pd5Q3irC+Neh/1AlmeBMVgZmP8yMLD8/c/cqv9r90whzv/MX7Eq/MfAwMDIwCuZdfSV8U8WDgZGRmYGrAoZGRgY/jO8b3sj/J2F6T8j4z80pzEhmIwMjAxsSbqqlkeZGP//Z8SlkJnhPwMjwx/Guoe1NhmRwk+YGH5jV8jOwMPHzcDBysAwh8FrxQwtPU99HrwBXsnAwMDAsJiBgYGBoZ1xmKYqALHhMpn1o7igAAAAAElFTkSuQmCC"></button>
    <button onclick="animGXFDZGNIPCQHGVWJ.reverse_animation()"><img class="anim_icon" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAQAAAAngNWGAAAAAXNSR0IArs4c6QAAAAJiS0dEAP+Hj8y/AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3QURCAgmVvZElgAAAVFJREFUKM+t0k8ow3EYx/H3s/2aLDUSZctFkgsHEi1XLi5ukpPSWsuJklwclsPSsDKFi7MSJ0I5qF2GHO2m0FY7+BdNv7Y9DpuxDSt5vsfvq+fT9/k+8D8VBxIAWH6H0ead4Qb5BRwCENoceZi5Stl/6BgCBmtWhjzxg4mUQ02rAhil7JgB9tze7aTLxFAKsUUd14B9ZzCyFUk401gQyQJaDNcBHwv7t7ETd0ZVQFEEzcNCdE/1wtj15imGWlEB8qkf2QaAWjbG/bPSamIDyX65/iwDIFx7tWjUvWCoSo5oGbYATN7PORt7W9IZEQXJH8ohuN7C0VVX91KNqYhq4a1lEGJI0j892tazXCWQRUpwAbYDcHczPxXuajq3mbnhfANz5eOJxsuNvs7+jud0UcuyL3QAkuEMx4rnIvBYq1JhEwPAUb3fG7x8tVdc292/7Po7f2VqA+Yz7ZwAAAAASUVORK5CYII="></button>
    <button onclick="animGXFDZGNIPCQHGVWJ.pause_animation()"><img class="anim_icon" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAQAAAAngNWGAAAAAXNSR0IArs4c6QAAAAJiS0dEAP+Hj8y/AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3QURCAkR91DQ2AAAAKtJREFUKM9jYCANTEVib2K4jcRbzQihGWEC00JuNjN8Z2Q0Zo3VYWA4lL005venH9+c3ZK5IfIsMIXMBtc12Bj+MMgxMDAwMPzWe2TBzPCf4SLcZCYY4/9/RgZGBiaYFf8gljFhKiQERhUOeoX/Gf8y/GX4y/APmlj+Mfxj+MfwH64Qnnq0zr9fyfLrPzP3eQYGBobvk5x4GX4xMIij23gdib0cRWYHiVmAAQDK5ircshCbHQAAAABJRU5ErkJggg=="></button>
    <button onclick="animGXFDZGNIPCQHGVWJ.play_animation()"><img class="anim_icon" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAQAAAAngNWGAAAAAXNSR0IArs4c6QAAAAJiS0dEAP+Hj8y/AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3QURCAkEmo00MwAAAS9JREFUKM+tkj1IQmEUhp9j94LQj0FD4RRBLdLQ3ftb26PRcCiQIIiIDFwKC0OhaAiam5wVDBpqCKohQojMLYzaAiUatOtpuQrKVQl64fu+4Xt4OLwc+Fs+nNM16jsPAWS6gZXggoZfXmfhog3hcZ6aTXF87Sp68OmH4/YggAo8bmfyyeh6Z1AAKPVldyO1+Iz2uILq3AriJSe3l+H7aj+cuRnrTsVDxSxay+VYbMDnCtZxxQOU9G4nlU9E1HQBxRkCQMRGRnIbpxMARkvxCIoAorYMMrq0mJ0qu4COUW3xyVDqJC4P+86P0ewDQbQqgevhlc2C8ETApXAEFLzvwa3EXG9BoIE1GQUbv1h7k4fTXxBu6cKgUbX5M3ZzNC+a7rQ936HV56SlRpcle+Mf8wvgJ16zo/4BtQAAAABJRU5ErkJggg=="></button>
    <button onclick="animGXFDZGNIPCQHGVWJ.next_frame()"><img class="anim_icon" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAQAAAAngNWGAAAAAXNSR0IArs4c6QAAAAJiS0dEAP+Hj8y/AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3QURCAkd/uac8wAAAMhJREFUKM9jYBie4DEUQ8B+fEq3+3UrMzAwMFxjYGBgYJizYubaOUxYFUaXh/6vWfRfEMIL/+//P5gZJoei4/f/7wxnY1PeNUXdE2RgYGZgYoCrY2BBVsjKwMDAwvCS4f3SG/dXxm5gYESSQ1HIwvCPgZmB8f8Pxv+Kxxb/YfiPJIdi9T8GJgaG/38ZFd4Fx0xUYsZt4h8GBgb2D2bLy7KnMTAwMEIxFoVCXIYr1IoDnkF4XAysqNIwUMDAwMDAsADKS2NkGL4AAIARMlfNIfZMAAAAAElFTkSuQmCC"></button>
    <button onclick="animGXFDZGNIPCQHGVWJ.last_frame()"><img class="anim_icon" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAQAAAAngNWGAAAAAXNSR0IArs4c6QAAAAJiS0dEAP+Hj8y/AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3QURCAknOOpFQQAAAS9JREFUKM/dkrEvQ3EQxz/33mtoQxiYpANbLU26NAabSCcSUouGBVNDjYQaOiDpIEiKjURIw2Kx04hEYmkHEpGoJpSISaXq9Wd4P03/ht5y98197/u9XA4aK4rAWw3lgWddZ3S+/G9mEovtAB8AHE4pgTQAx8PbJweRmsq6GimmNpxaNYXVzMNNCI6A2figimwCGACK786zuWgh3qcsKf/w0pM4X0m/doNVFVzVGlEQsdRj193VxEWpH0RsdRu+zi3tVMqCAsDShoiYqiSV4OouVDFEqS9Pbiyg7vV62lpQ2BJ4Gg0meg0MbNpkYG/e+540NNFyrE1a8qHk5BaAjfnrzUaHfAWImVrLIXbgnx4/9X06s35cweWsVACa3a24PVp0X+rPv1aHFnSONdiL8Qci0lzwpOM5sQAAAABJRU5ErkJggg=="></button>
    <button onclick="animGXFDZGNIPCQHGVWJ.faster()">+</button>
  <form action="#n" name="_anim_loop_selectGXFDZGNIPCQHGVWJ" class="anim_control">
    <input type="radio" name="state" value="once" > Once </input>
    <input type="radio" name="state" value="loop" checked> Loop </input>
    <input type="radio" name="state" value="reflect" > Reflect </input>
  </form>
</div>


<script language="javascript">
  /* Instantiate the Animation class. */
  /* The IDs given should match those used in the template above. */
  (function() {
    var img_id = "_anim_imgGXFDZGNIPCQHGVWJ";
    var slider_id = "_anim_sliderGXFDZGNIPCQHGVWJ";
    var loop_select_id = "_anim_loop_selectGXFDZGNIPCQHGVWJ";
    var frames = new Array(0);
    
  frames[0] = "data:image/png;base64,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"
  frames[1] = "data:image/png;base64,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fjnUU4p5f/nLX3TuuefqzTffVEFBgWbMmBHqKeGYmTNn6tlnnw31NGLekiVLVFtbq7fffluTJ09WQUFBqKcESa+88ooeeOAB1dXVhXoqOObDDz9U27Zt9be//U0vv/yyHn300VBPCQHGSYRB8tvf/lYNZ9bs3r1brVu3DvGM8Lvf/U7JycmSjq4etGjRIsQzQoPevXtr2LBheuedd0I9lZi2bt06nX/++ZKkXr16acOGDSGeESSpc+fOKiws1JQpU0I9FRxz6aWX6pJLLpEkWSwWJSYSr6IdTzgA5s+fr9dff91QKygoUM+ePfXb3/5W33//vV577bUQzS42uXsmZWVlmjJliu6///4QzS52uXoul156qdasWROiWaFBdXW10tPTbePExERZLBbFx/MfL0Np2LBh2rVrV6ingUZatWol6eifmbvuuksTJ04M8YwQaAToABgxYoRGjBjh9LPXX39dP/74o8aNG6fFixcHeWaxy9Uz2bJli+655x5NnTpVffv2DcHMYpu7PysIvbS0NJnNZtuY8Ay4VlJSogkTJuiGG27Qr371q1BPBwHGvwmD5KWXXtKCBQskSSkpKUpISAjxjLB161bdfffdmjVrls4777xQTwcIO71799aKFSskSV9//bW6desW4hmhsYa2QIReeXm5xo4dq3vvvVdXXnllqKeDIGAFOkiuvvpqTZ06VfPnz5fVauVlnDAwe/Zs1dbWaubMmbJarcrIyFBhYWGopwWEjWHDhunzzz/XqFGjJIl/b4WZuLi4UE8Bx8ydO1eVlZV64YUXVFhYqLi4OL3yyiu292wQfeKs/AoLAAAAeIwWDgAAAMALBGgAAADACwRoAAAAwAsEaAAAAMALBGgAAADACwRoAAAAwAsEaAAAAMALBGgAAADACwRoAAAAwAsEaAAAAMALBGgAAADACwRoAAAAwAsEaAAAAMALBGgAAADACwRoAAAAwAsEaAAAAMALBGgAAADACwRoAAAAwAsEaAAAAMALBGgAAADACwRoAAAAwAsEaAAAAMALBGgAAADACwRoAAAAwAsEaAAAAMALBGgAAADACwRoAAAAwAsEaAAAAMALBGgAAADACwRoAAAAwAsEaAAAAMALBGgAAADACwRoAAAAwAsEaAAAAMALBGgAAADACwRoAAAAwAsEaAAAAMALBGgAAADACwRoAAAAwAv/DyyiaywNYN7hAAAAAElFTkSuQmCC"
  frames[2] = "data:image/png;base64,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+s3y+3Tl219oRMlq1/ieb3fImUVPMAAAVkGABixg96E9enjl43rinvnqsn2fpBNb1H3yuZSUFBOnCgIAkCgI0IDJVpWvUekXr6r0phddY47JU3Vo8tTjr4fhVEECOAAA4UOABkz056/mqXL1cr1y319dYwcWvaea8y9wXYd63HWsHOsNAECsIEADJqipO6a7lt+nX/3jK93/yqeu8T2b/ivnSe3c3hvqqYKhBnAAAOCOAA1EWcUhux759xP63wff0ulbKiRJtSfnaN+6TVJSktf7Qz3uOlaO9QYAIFZwlDcQRSvLP9fjy3+r0mHPusLzoTvu1r6vNvsMz1Lox13HyrHeAADEClaggSiZs+4l/fDlx3p1ynzX2IEFi1Qz4OcBPxfqqYKxcqw3AACxggANRFhNbY3uWnG/Bi3ZoGkv/Ms1vmfD/8lpswV1j1BOFYylY70BAIgFBGgPS5Ys0XvvvaennnpKkrR06VI9/vjjysnJkSTdeeed6tu3r2bPnq0VK1YoNTVVU6dOVc+ePc0sGxb1vaNCj656Sg/MeEc9v94pSarLaq29/9kmJUevg4pjvQEACB8CdAMzZszQJ598om7durnG1q9frylTpmjQoEGusY0bN+rzzz/X/PnzVV5erjvuuEMLFiwwo2RY2Ke7Vqv0qxKVjnreNXbolrFy/PYJE6sCAAChIkA30Lt3bw0aNEhvvvmma2zDhg3atGmTXn75ZfXs2VOTJk3SmjVr1L9/f0lSTk6O6urqtH//frVt29as0mExxWtf0IGv/63XJ/7479IPf5mv6oGDTawKAACEQ0IG6AULFmjevHluYzNnztRll12m1atXu433799fAwcOVG5urqZPn66SkhJVVVW5heW0tDSvMSSm6toa3b3ifl28fJMeena5a3zvuk2qy+loXmEAACBsEjJADx06VEOHDg3qvVdffbUyMzMlSZdcconef/99devWTVVVVa73OBwO13sCsdkafw+iK5xzsuOHXZr43qOa/OR7OvfzrZIkZ0qKko4eVbuUlLB9TyLgz4r1MCfWxLxYD3OSGBIyQBsxZMgQlZSUKDs7W//+97911llnqWfPnnryySc1evRolZeXy+l0qk2bNo3ey24/GIWKESybLTNsc/Lxzn+rdMN8lV7/nGvs8HWjVDVrtrTP90Em8C2c84LwYE6siXmxnniaE34QCIwA3YgZM2Zo/PjxatmypU477TQVFhYqJSVFffr00bXXXiun06lp06aZXSZM9Mcv/6wD36zRG3e94Rr74cXXVH35EBOrAgAAkZLkdDqdZheRKOLlp9J4EepKQXVtte5e8YDyP/1Wd/9xqWt875r1quvUORwlJqR4WsGJF8yJNTEv1hNPc8IKdGCsQANNsKvqe81Y/bTumP2BLvp4i2vcvnOv1KyZiZUBAIBII0ADBn1UtlLzv/mrSq/7s2vsyFVX6+Dcl0ysKvJWbazQ4pVbtWvPIXVsn6aC/DwOZwEAJCQCNGDA7794Vvu+XaeS8a+7xiqffUFHf32NiVVF3qqNFZq7aIPruszucF0TogEAiSZ6ZwkDMexobbXGLZuiNks+0JwG4Xnvv7+M+/AsSYtXbvUzvi2qdQAAYAWsQAON2FlVrt+unqUxf16hgcu+cY3bd9ilFi1MrCx6du3xvRVf+V5HlCsBAMB8BGgggOU7PtGCzW/pLyOfV2ptnSTp6ODLVPnqm418Mr50bJ+mMrt3WM5pl25CNQAAmIsWDsCPJz8v1tJVf9GbI/7sCs+Vf3gm4cKzJBXk5/kZ7xLdQgAAsABWoAEPR44d1cSPHlSvtdv1+GP/cI3v+/gz1Z5+homVmaf+QcHFK7epfK9DOe3SVZDfhQcIAQAJiQANNLDj4E499tkfdMO8T1Tw7teucfu2CqlVKxMrM1+/7tkEZgAARIAGXJbt+Jf+tnmRXrz1JWU4qiVJ1RddrB/++o7JlQEAACshQAOSfvfZn7S/7D96c+w819jBx57SkZtvNbEqAABgRQRoJLQjx45o4kfT1GP9Tj3xvz+uNO/78FPV9jjLxMoAAIBVEaCRsL7bt033fvSYhr+xSlct/NI1bv9vuZTO9mwAAMA3AjQS0tLtK/TWlr/rmfGvq/3eKklSTe8+OvDehyZXBgAArI4AjYTidDr1+Gd/0L7vv1PpLS+7xqsemqHDt99hXmEAACBmEKCRMA4fO6JJH03TGZvL9dT0ha7x/e8v17Gf9TaxMgAAEEuSnE6n0+wiAAAAgFjBUd4AAACAAQRoAAAAwAACNAAAAGAAARoAAAAwgAANAAAAGECABgAAAAwgQEfJ4cOHdfvtt+v666/XzTffrN27d5tdUsKrqqpSUVGRRo4cqWHDhmnt2rVml4QGlixZookTJ5pdRkJzOp2aPn26hg0bplGjRmnHjh1ml4QT1q1bp5EjR5pdBk44duyYpkyZouuuu06FhYVatmyZ2SUhwgjQUVJaWqqzzjpLr732mq644go999xzZpeU8F566SVdcMEFevXVVzVz5kw98sgjZpeEE2bMmKFZs2aZXUbCW7p0qaqrq1VSUqKJEydq5syZZpcESc8//7weeOAB1dTUmF0KTli0aJHatm2r119/Xc8995weffRRs0tChHESYZTccMMNqj+zZteuXWrdurXJFeGmm25S8+bNJR1fPWjRooXJFaFe7969NWjQIL355ptml5LQ1qxZo4suukiS1KtXL61fv97kiiBJXbp0UXFxsaZMmWJ2KTjhsssu06WXXipJqqurU2oq8SreMcMRsGDBAs2bN89tbObMmTrrrLN0ww03aMuWLXrxxRdNqi4xBZoTu92uKVOm6P777zepusTlb14uu+wyrV692qSqUK+qqkqZmZmu69TUVNXV1Sk5mf94aaZBgwZp586dZpeBBlq1aiXp+J+Z3/zmN7r77rtNrgiRRoCOgKFDh2ro0KE+X5s3b56+++47jR07VkuWLIlyZYnL35xs3rxZkyZN0j333KO+ffuaUFliC/RnBebLyMiQw+FwXROeAf/Ky8s1fvx4XX/99frVr35ldjmIMP4mjJI///nPWrhwoSQpLS1NKSkpJleEb7/9VnfddZeefPJJXXjhhWaXA1hO7969tWLFCknS2rVrdfrpp5tcERqqbwuE+fbs2aPRo0dr8uTJuuqqq8wuB1HACnSUXH311brnnnu0YMECOZ1OHsaxgKefflrV1dWaMWOGnE6nsrKyVFxcbHZZgGUMGjRIn3zyiYYNGyZJ/L1lMUlJSWaXgBPmzp2ryspKPfPMMyouLlZSUpKef/5513M2iD9JTn6EBQAAAIJGCwcAAABgAAEaAAAAMIAADQAAABhAgAYAAAAMIEADAAAABhCgAQAAAAMI0AAAAIABBGgAAADAAAI0AAAAYAABGgAAADCAAA0AAAAYQIAGAAAADCBAAwAAAAYQoAEAAAADCNAAAACAAQRoAAAAwAACNAAAAGAAARoAAAAwgAANAAAAGECABgAAAAwgQAMAAAAGEKABAAAAAwjQAAAAgAEEaAAAAMAAAjQAAABgAAEaAAAAMIAADQAAABhAgAYAAAAMIEADAAAABhCgAQAAAAMI0AAAAIABBGgAAADAAAI0AAAAYAABGgAAADCAAA0AAAAYQIAGAAAADCBAAwAAAAYQoAEAAAADCNAAAACAAf8fN3K8B71zB9YAAAAASUVORK5CYII="


    /* set a timeout to make sure all the above elements are created before
       the object is initialized. */
    setTimeout(function() {
        animGXFDZGNIPCQHGVWJ = new Animation(frames, img_id, slider_id, 100, loop_select_id);
    }, 0);
  })()
</script>




Remember that the linear regression cost function is convex, and more precisely quadratic. We can see the path that gradient descent takes in arriving at the optimum:



```python
from mpl_toolkits.mplot3d import Axes3D

def error(X, Y, THETA):
    return np.sum((X.dot(THETA) - Y)**2)/(2*Y.size)

ms = np.linspace(theta[0] - 20 , theta[0] + 20, 20)
bs = np.linspace(theta[1] - 40 , theta[1] + 40, 40)

M, B = np.meshgrid(ms, bs)

zs = np.array([error(xaug, y, theta) 
               for theta in zip(np.ravel(M), np.ravel(B))])
Z = zs.reshape(M.shape)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(M, B, Z, rstride=1, cstride=1, color='b', alpha=0.2)
ax.contour(M, B, Z, 20, color='b', alpha=0.5, offset=0, stride=30)


ax.set_xlabel('Intercept')
ax.set_ylabel('Slope')
ax.set_zlabel('Cost')
ax.view_init(elev=30., azim=30)
ax.plot([theta[0]], [theta[1]], [cost[-1]] , markerfacecolor='r', markeredgecolor='r', marker='o', markersize=7);
#ax.plot([history[0][0]], [history[0][1]], [cost[0]] , markerfacecolor='r', markeredgecolor='r', marker='o', markersize=7);


ax.plot([t[0] for t in history], [t[1] for t in history], cost , markerfacecolor='r', markeredgecolor='r', marker='.', markersize=2);
ax.plot([t[0] for t in history], [t[1] for t in history], 0 , markerfacecolor='r', markeredgecolor='r', marker='.', markersize=2);
```



![png](gradientdescent_files/gradientdescent_18_0.png)


## Stochastic gradient descent

As noted, the gradient descent algorithm makes intuitive sense as it always proceeds in the direction of steepest descent (the gradient of $J$) and guarantees that we find a local minimum (global under certain assumptions on $J$). When we have very large data sets, however, the calculation of $\nabla (J(\theta))$ can be costly as we must process every data point before making a single step (hence the name "batch"). An alternative approach, the stochastic gradient descent method, is to update $\theta$ sequentially with every observation. The updates then take the form:

$$\theta := \theta - \alpha \nabla_{\theta} J_i(\theta)$$

This stochastic gradient approach allows us to start making progress on the minimization problem right away. It is computationally cheaper, but it results in a larger variance of the loss function in comparison with batch gradient descent. 

Generally, the stochastic gradient descent method will get close to the optimal $\theta$ much faster than the batch method, but will never fully converge to the local (or global) minimum. Thus the stochastic gradient descent method is useful when we want a quick and dirty approximation for the solution to our optimization problem. A full recipe for stochastic gradient descent follows:

- Initialize the parameter vector $\theta$ and set the learning rate $\alpha$
- Repeat until an acceptable approximation to the minimum is obtained:
    - Randomly reshuffle the instances in the training data.
    - For $i=1,2,...m$ do: $\theta := \theta - \alpha \nabla_\theta J_i(\theta)$
    
The reshuffling of the data is done to avoid a bias in the optimization algorithm by providing the data examples in a particular order. In code, the algorithm should look something like this:

```python
for i in range(nb_epochs):
  np.random.shuffle(data)
  for example in data:
    params_grad = evaluate_gradient(loss_function, example, params)
    params = params - learning_rate * params_grad
```

For a given epoch, we first reshuffle the data, and then for a single example, we evaluate the gradient of the loss function and then update the params with the chosen learning rate.

The update for linear regression is:

$$\theta_j := \theta_j + \alpha (y^{(i)}-f_\theta (x^{(i)})) x_j^{(i)}$$




```python
def sgd(x, y, theta_init, step=0.001, maxsteps=0, precision=0.001, ):
    costs = []
    m = y.size # number of data points
    oldtheta = 0
    theta = theta_init
    history = [] # to store all thetas
    preds = []
    grads=[]
    counter = 0
    oldcost = 0
    epoch = 0
    i = 0 #index
    pred = np.dot(x[i,:], theta)
    error = pred - y[i]
    gradient = x[i,:].T*error
    grads.append(gradient)
    print(gradient,x[i],y[i],pred, error, np.sum(error ** 2) / 2)
    currentcost = np.sum(error ** 2) / 2
    counter+=1
    preds.append(pred)
    costsum = currentcost
    costs.append(costsum/counter)
    history.append(theta)
    print("start",counter, costs, oldcost)
    while 1:
        #while abs(costs[counter-1] - oldcost) > precision:
        #while np.linalg.norm(theta - oldtheta) > precision:
        #print("hi", precision)
        #oldcost=currentcost
        gradient = x[i,:].T*error
        grads.append(gradient)
        oldtheta = theta
        theta = theta - step * gradient  # update
        history.append(theta)
        i += 1
        if i == m:#reached one past the end.
            #break
            epoch +=1
            neworder = np.random.permutation(m)
            x = x[neworder]
            y = y[neworder]
            i = 0
        pred = np.dot(x[i,:], theta)
        error = pred - y[i]
        currentcost = np.sum(error ** 2) / 2
        
        #print("e/cc",error, currentcost)
        if counter % 25 == 0: preds.append(pred)
        counter+=1
        costsum += currentcost
        oldcost = costs[counter-2]
        costs.append(costsum/counter)
        #print(counter, costs, oldcost)
        if maxsteps:
            #print("in maxsteps")
            if counter == maxsteps:
                break
        
    return history, costs, preds, grads, counter, epoch
```




```python
history2, cost2, preds2, grads2, iters2, epoch2 = sgd(xaug, y, theta_i, maxsteps=5000, step=0.01)

```


    [-24.75029678  -0.79828189] [ 1.          0.03225343] 11.5348518902 -13.2154448901 -24.7502967803 306.288595356
    start 1 [306.2885953559632] 0




```python
print(iters2, history2[-1], epoch2, grads2[-1])
```


    5000 [ -3.57506492  82.86543911] 49 [-35.84384141  35.89692535]




```python
plt.plot(range(len(cost2[-10000:])), cost2[-10000:], alpha=0.4);
plt.xlim
```





    <function matplotlib.pyplot.xlim>




![png](gradientdescent_files/gradientdescent_23_1.png)




```python
from mpl_toolkits.mplot3d import Axes3D

def error(X, Y, THETA):
    return np.sum((X.dot(THETA) - Y)**2)/(2*Y.size)

ms = np.linspace(theta[0] - 20 , theta[0] + 20, 20)
bs = np.linspace(theta[1] - 40 , theta[1] + 40, 40)

M, B = np.meshgrid(ms, bs)

zs = np.array([error(xaug, y, theta) 
               for theta in zip(np.ravel(M), np.ravel(B))])
Z = zs.reshape(M.shape)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(M, B, Z, rstride=1, cstride=1, color='b', alpha=0.1)
ax.contour(M, B, Z, 20, color='b', alpha=0.5, offset=0, stride=30)


ax.set_xlabel('Intercept')
ax.set_ylabel('Slope')
ax.set_zlabel('Cost')
ax.view_init(elev=30., azim=30)
#ax.plot([theta[0]], [theta[1]], [cost[-1]] , markerfacecolor='r', markeredgecolor='r', marker='o', markersize=7);
#ax.plot([history[0][0]], [history[0][1]], [cost[0]] , markerfacecolor='r', markeredgecolor='r', marker='o', markersize=7);


#ax.plot([t[0] for t in history2], [t[1] for t in history2], cost2 , markerfacecolor='r', markeredgecolor='r', marker='.', markersize=2);
ax.plot([t[0] for t in history2], [t[1] for t in history2], 0 , alpha=0.5, markerfacecolor='r', markeredgecolor='r', marker='.', markersize=2);
```



![png](gradientdescent_files/gradientdescent_24_0.png)




```python
plt.plot([t[0] for t in history2], [t[1] for t in history2],'o-', alpha=0.1)
```





    [<matplotlib.lines.Line2D at 0x1236855f8>]




![png](gradientdescent_files/gradientdescent_25_1.png)




```python
grads2[-20:]
```





    [array([-27.82199581,  23.58326418]),
     array([ 16.43526198,  19.74364665]),
     array([ 8.1891872 , -1.63239905]),
     array([-12.71915021,  -0.41023618]),
     array([ 2.99788693,  0.06548086]),
     array([ 10.25001845,  -4.94598386]),
     array([-1.33529948, -1.86053324]),
     array([ 12.16237847,   9.88235005]),
     array([-16.17377054,  -8.17445577]),
     array([ 16.41555128,   7.65473473]),
     array([ 11.02368605, -24.28169329]),
     array([ 9.32730843, -6.75920199]),
     array([ 29.31098914,   4.35305735]),
     array([-4.94262659,  4.74037539]),
     array([-33.79289273, -15.03180404]),
     array([ 31.12316292,   0.97513998]),
     array([-13.48642442, -12.08044909]),
     array([-22.83846195, -20.18044723]),
     array([ -8.32998184, -11.42992274]),
     array([-35.84384141,  35.89692535])]





```python
history2[-20:]
```





    [array([ -3.5973351 ,  82.63068495]),
     array([ -3.76168772,  82.43324848]),
     array([ -3.84357959,  82.44957247]),
     array([ -3.71638809,  82.45367483]),
     array([ -3.74636696,  82.45302002]),
     array([ -3.84886714,  82.50247986]),
     array([ -3.83551415,  82.52108519]),
     array([ -3.95713793,  82.42226169]),
     array([ -3.79540023,  82.50400625]),
     array([ -3.95955574,  82.4274589 ]),
     array([ -4.0697926 ,  82.67027584]),
     array([ -4.16306569,  82.73786786]),
     array([ -4.45617558,  82.69433728]),
     array([ -4.40674931,  82.64693353]),
     array([ -4.06882038,  82.79725157]),
     array([ -4.38005201,  82.78750017]),
     array([ -4.24518777,  82.90830466]),
     array([ -4.01680315,  83.11010913]),
     array([ -3.93350333,  83.22440836]),
     array([ -3.57506492,  82.86543911])]



## Mini-batch gradient descent
What if instead of single example from the dataset, we use a batch of data examples witha given size every time we calculate the gradient:

$$\theta = \theta - \eta \nabla_{\theta} J(\theta; x^{(i:i+n)}; y^{(i:i+n)})$$

This is what mini-batch gradient descent os about. Using mini-batches has the advantage that the variance in the loss function is reduced, while the computational burden is still reasonable, since we do not use the full dataset. The size of the mini-batches becomes another hyper-parameter of the problem. In standard implementations it ranges from 50 to 256. In code, mini-batch gradient descent looks like this:

```python
for i in range(nb_epochs):
  np.random.shuffle(data)
  for batch in get_batches(data, batch_size=50):
    params_grad = evaluate_gradient(loss_function, batch, params)
    params = params - learning_rate * params_grad
```

The difference with SGD is that for each update we use a batch of 50 examples to estimate the gradient.

## Variations on a theme

### Momentum

Often, the cost function has ravines near local optima, ie. areas where the shape of the function is significantly steeper in certain dimensions than in others. This migh result in a slow convergence to the optimum, since standard gradient descent will keep oscillating about these ravines. In the figures below, the left panel shows convergence without momentum, and the right panel shows the effect of adding momentum:

<table><tr><td><img src="http://sebastianruder.com/content/images/2015/12/without_momentum.gif", width=300, height=300></td><td><img src="http://sebastianruder.com/content/images/2015/12/with_momentum.gif", width=300, height=300></td></tr></table>

One way to overcome this problem is by using the concept of momentum, which is borrowed from physics. At each iteration, we remember the update $v = \Delta \theta$ and use this *velocity* vector (which as the same dimension as $\theta$) in the next update, which is constructed as a combination of the cost gradient and the previous update:

$$v_t = \gamma v_{t-1} +  \eta \nabla_{\theta} J(\theta)$$
$$\theta = \theta - v_t$$

The effect of this is the following: the momentum terms increases for dimensions whose gradients point in the same direction, and reduces the importance of dimensions whose gradients change direction. This avoids oscillations and improves the chances of rapid convergence. The concept is analog to the  a rock rolling down a hill: the gravitational field (cost function) accelerates the particule (weights vector), which accumulates momentum, becomes faster and faster and tends to keep travelling in the same direction. A commonly used value for the momentum parameter is $\gamma = 0.5$.
