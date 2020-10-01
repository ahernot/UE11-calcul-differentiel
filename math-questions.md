#### <b>Question 1</b>

Soit $c \in \mathbb{R}$.
On suppose que la fonction $f:\mathbb{R}^2 \to \mathbb{R}$ est continue et vérifie
$f(x_1, x_2) \to +\infty$ quand $\|(x_1,x_2)\| \to +\infty$.
Que peut-on dire de l'ensemble de niveau $c$ de $f$ ?
<br><br>
<b>IDÉE :</b>
<br>
On définit l'ensemble de niveau $c$ :<br>
$$\mathrm{LVL}(c) = \left\{ (x_1, x_2) \mid f(x_1,x_2) = c \right\}$$

Montrons que l'ensemble de niveau $c$ de $f$ est vide ou borné. (???)<br>
$f(x_1,x_2) \xrightarrow[\|(x_1,x_2)\| \to +\infty]{} +\infty$. Alors $\exists M(c) > 0, \ \|(x_1,x_2)\| \geq M(c) \ \Rightarrow \ f(x_1,x_2) > c$.<br>
Donc $\mathrm{LVL}(c) \subset \left\{(x_1, x_2) \mid \|(x_1,x_2)\| < M(c) \right\} = E_M(c)$, un ensemble borné.

Montrer que $f$ admet forcément un minimum sur $E_M(c)$. Si ce min est >≠ c, alors LVL(c) est $\emptyset$.

<br><br><br>

Dans la suite la fonction $f$ est supposée continûment différentiable. On suppose également que le gradient $\nabla f$ ne s'annule pas dans un voisinage du point $x_0 = (x_{10}, x_{20}) \in \mathbb{R}^2$. On pose alors
$$
p(x_1, x_2) := \frac{\partial_2 f(x_0)}{\|\nabla f(x_0)\|} (x_1 - x_{10}) -
\frac{\partial_1 f(x_0)}{\|\nabla f(x_0)\|} (x_2 - x_{20}).
$$
#### <b>Question 2</b>
Comment interpréter géométriquement le terme $p(x_1,x_2)$ ?
<br><br>
<b>IDÉE :</b>
