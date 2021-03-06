--- 
title: "Interprétabilité de boîtes noires"
author: "Lucie Guillaumin & Mehdi Chebli"
date: "2021-01-04"
site: bookdown::bookdown_site
documentclass: book
bibliography: [book.bib, packages.bib]
biblio-style: apalike
link-citations: yes
geometry:
  - top=30mm
  - left=20mm
  - heightrounded
lang: fr
description: "Projet tutoré présenté par Lucie Guillaumin et Mehdi Chebli encadré par Mr. Perduca."
---

#Introduction 
![](logomemoire.png)
  
Le machine learning (apprentissage automatique en français) est une technologie d’intelligence artificielle, qui à
l’aide d’approches mathématiques et surtout statistiques permet de donner la capacité aux machines d’apprendre
à partir des données. C’est-à-dire d’améliorer leurs performances à résoudre des tâches sans être explicitement
programmés pour chacunes.  
L'objectif de ces algorithmes consiste à estimer un modèle à partir des observations puis prédire une variable que l'on souhaite expliquer.  
  
Certains de ces modèles de machine learning sont appelés boîtes noires. En effet, après application du modèle, on obtient des résultats mais nous ne savons pas comment ni pourquoi la machine est parvenue à ce résultat.  
Dans ce mémoire, nous cherchons donc à expliquer et à comprendre comment et pourquoi un modèle de boite noire donne tel ou tel résultat. Les différents chapitres que contiennent ce livre expliqueront donc comment résoudre ce problème appelé interprétabilité.   

On peut trouver différentes définitions de l’interprétabilité sur le net, celles que nous retiendrons sont les suivantes :  

-	la capacité dans laquelle un être humain peut comprendre la cause d’une décision  

-	le degré auquel un être humain peut prédire de manière cohérente le résultat d’un modèle  

L’interprétabilité est très importante dans le monde des modèles de machine learning. En réalité, dans de nombreux domaines il faut expliquer/justifier une prise de décision provoquée par le modèle prédictif. Notons qu’elle n’est pas requise si le modèle étudié n’a pas d’impact significatif : si le modèle possède un impact social ou financier, l’interprétabilité devient alors pertinente. De plus, lorsque le modèle est bien étudié il n’est pas nécessaire de faire appel à elle.  

Certains algorithmes disposent de méthodes permettant de déterminer l’importance des variables, ils n’indiquent cependant pas si une variable affecte positivement ou négativement le modèle. 
Effectivement, dans la plupart des cas, on ne se soucie généralement pas de savoir pourquoi telle ou telle décision a été prise : on ne cherche qu’à savoir si la performance prédictive sur un jeu de données de test est correct. Mais il faut néanmoins faire attention : une prédiction correcte ne résout que partiellement notre problème initial, elle ne nous donne aucune information ni explication.  

Le besoin d’interprétabilité signifie donc qu’il ne suffit pas d‘obtenir la prédiction (le quoi) mais plutôt que le modèle doit expliquer comment il en est arrivé à la prédiction (le pourquoi). Mais encore une fois attention, les explications du modèle ne doivent pas expliquer entièrement le déroulement du modèle, ils doivent plutôt aborder une ou deux cause(s) principale(s).  

Pour cela, nous avons choisi trois méthodes : les graphiques de dépendance partielle, la permutation de variable afin de remarquer les variables les plus importantes dans les modèles prédictif, et la méthode LIME.   
Nous nous appuierons donc sur des exemples appliqués à des ensembles de données ainsi que sur des modèles de boites noires présentés dans les références du livre.  
  
Nous nous baserons sur le livre `Interpretable Machine Learning` de Christophe Molnar.  

##Remerciements {-}
Nous tenons à remercier Mr. Perduca qui nous a permis de bénéficier de son encadrement.  
Les conseils qu’il nous a prodigué, la patience, la confiance qu’il nous a témoignés ont été déterminants dans la réalisation de notre travail.  
Nos remerciements s’étendent également à tous nos enseignants durant les années des études.
Enfin, nous tenons à remercier tous ceux qui, de près ou de loin, ont contribué à la réalisation de ce travail.

<img src="logouniv.png" width="930" style="display: block; margin: auto;" />
