### <b>Question 1</b>

Soit $c \in \mathbb{R}$.
On suppose que la fonction $f:\mathbb{R}^2 \to \mathbb{R}$ est continue et vérifie
$f(x_1, x_2) \to +\infty$ quand $\|(x_1,x_2)\| \to +\infty$.
Que peut-on dire de l'ensemble de niveau $c$ de $f$ ?
<br><br>
<b>IDÉE :</b>
<br>
On définit l'ensemble de niveau $c$ :<br>
$$\mathrm{LVL}(c) = \left\{ (x_1, x_2) \mid f(x_1,x_2) = c \right\}$$

Montrons que l'ensemble de niveau $c$ de $f$ est borné. (???)<br>
$f(x_1,x_2) \xrightarrow[\|(x_1,x_2)\| \to +\infty]{} +\infty$. Alors $\exists M(c) > 0, \ \|(x_1,x_2)\| \geq M(c) \ \Rightarrow \ f(x_1,x_2) > c$.<br>
Donc $\mathrm{LVL}(c) \subset \left\{(x_1, x_2) \mid \|(x_1,x_2)\| < M(c) \right\} = E_M(c)$, un ensemble borné.

Montrons que l'ensemble de niveau $c$ de $f$ est fermé

=> dans $\mathbb{R}^n$, c'est un compact (???).


???? Montrer que $f$ admet forcément un minimum sur $E_M(c)$. Si ce min est >≠ c, alors LVL(c) est $\emptyset$.

<br><br><br>

Dans la suite la fonction $f$ est supposée continûment différentiable. On suppose également que le gradient $\nabla f$ ne s'annule pas dans un voisinage du point $x_0 = (x_{10}, x_{20}) \in \mathbb{R}^2$. On pose alors
$$
p(x_1, x_2) := \frac{\partial_2 f(x_0)}{\|\nabla f(x_0)\|} (x_1 - x_{10}) -
\frac{\partial_1 f(x_0)}{\|\nabla f(x_0)\|} (x_2 - x_{20}).
$$

<br>


### <b>Question 2</b>
Comment interpréter géométriquement le terme $p(x_1,x_2)$ ?
<br><br>
<b>IDÉE :</b>
Ça semble être une sorte d'indicateur de ??????????

C'est un plan

<br><br><br>
En changeant légèrement les notations :
<br>

$$
p(x, y) = \frac{1}{\| \nabla f(X_0) \|} \left( \frac{\partial f}{\partial y}(X_0) (x - x_0) - \frac{\partial f}{\partial x}(X_0) (y - y_0) \right)
$$


ie la proportion du gradient suivant y * ∆x - la proportion du gradient suivant x * ∆y

ie :

$$
p(x, y)
= 
\begin{vmatrix}
\text{prop grad}\ y & \Delta y
\\
\text{prop grad}\ x & \Delta x
\end{vmatrix}
=
\frac{1}{\| \nabla f(X_0) \|}
\begin{vmatrix}
\partial_y f (x_0, y_0) & y-y_0
\\
\partial_x f (x_0, y_0) & x-x_0
\end{vmatrix}
$$

Notons $K_\mathrm{grad} = \frac{1}{\| \nabla f(X_0) \|}$.

$p(x_0, y_0) = 0$

$p_{y_0} : x \mapsto p(x, y_0) = \partial_y f (x_0, y_0) K_\mathrm{grad} \cdot (x-x_0)$ est une fonction affine

$p_{x_0} : y \mapsto p(x_0, y) = - \partial_x f (x_0, y_0) K_\mathrm{grad} \cdot (y-y_0)$ est une fonction affine



<br><br><br><br><br><br><br><br>
### <b>Question 5</b>
L'application à laquelle nous destinons la fonction `Newton` demande-t-elle une grande précision ?
Choisir une valeur de `eps` qui semble raisonnable et justifier l'ordre de grandeur choisi.

Choix de $\varepsilon$ : __PLACEHOLDER__
Justification : __PLACEHOLDER__

<br><br>

#### Tâche 2

Testez votre implémentation de la fonction `Newton` ! On suggère par exemple de l'utiliser pour chercher un point $(x_1, x_2)$ de la ligne de niveau $0.8$ de $f_1$ (cf. Exemples de référence) qui vérifie en outre $x_1 = x_2$ en utilisant le point initial $(0.8, 0.8)$. Puis de faire varier le point initial, la contrainte supplémentaire, etc. et de représenter graphiquement les résultats.

__TO PROGRAM__





