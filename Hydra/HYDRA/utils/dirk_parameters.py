"""
Classe définissant les paramètres d'un schéma DIRK.
Stocke les coefficients de Butcher et fournit des méthodes utilitaires.

Cette classe implémente de nombreuses méthodes DIRK optimisées basées sur
l'article "Diagonally implicit Runge–Kutta methods for stiff ODEs" 
par Kennedy & Carpenter (Applied Numerical Mathematics 146, 2019).
"""
import numpy as np

class DIRKParameters:
    """
    Classe définissant les paramètres d'un schéma DIRK.
    Stocke les coefficients de Butcher et fournit des méthodes utilitaires.
    
    Cette classe implémente de nombreuses méthodes DIRK optimisées basées sur
    l'article "Diagonally implicit Runge–Kutta methods for stiff ODEs" 
    par Kennedy & Carpenter (Applied Numerical Mathematics 146, 2019).
    """
    def __init__(self, method="SDIRK2"):
        """
        Initialise les paramètres DIRK pour la méthode spécifiée.
        
        Parameters
        ----------
        method : str, optional
            Nom du schéma DIRK à utiliser. Options:
            - Méthodes traditionnelles : "SDIRK2", "SDIRK3", "ESDIRK3", "ESDIRK4"
            - Méthodes optimisées ordre 3 : "ESDIRK3(2)4L[2]SA", "ESDIRK3(2)5L[2]SA"
            - Méthodes optimisées ordre 4 : "ESDIRK4(3)6L[2]SA1", "ESDIRK4(3)6L[2]SA2", "ESDIRK4(3)7L[2]SA"
            - Méthodes optimisées ordre 5 : "ESDIRK5(4)7L[2]SA1", "ESDIRK5(4)7L[2]SA2", "ESDIRK5(4)8L[2]SA"
            - Méthodes optimisées ordre 6 : "ESDIRK6(5)9L[2]SA"
            
            Par défaut "SDIRK2".
            
        Note: La nomenclature p(q)sL[r]SA signifie:
            - p = ordre du schéma principal
            - q = ordre du schéma embarqué
            - s = nombre d'étapes
            - L = L-stabilité (stabilité pour valeurs propres → -∞)
            - r = ordre des étages (stage-order)
            - SA = stiffly-accurate (γ est répété dans la dernière ligne/colonne)
        """
        # Méthodes DIRK traditionnelles
        if method == "SDIRK2":
            # SDIRK d'ordre 2 avec gamma = 1 - 1/sqrt(2)
            gamma = 1.0 - 1.0/np.sqrt(2.0)
            self.A = np.array([[gamma, 0.0], 
                               [1.0-gamma, gamma]])
            self.c = np.array([gamma, 1.0])
            self.b = np.array([1.0-gamma, gamma])
            self.order = 2
            self.embedded_order = None
            self.bhat = None
            
        elif method == "SDIRK3":
            # SDIRK d'ordre 3
            gamma = 0.4358665215084589994160194  # Racine de x³ - 3x² + 3x - 1/2 = 0
            self.A = np.array([
                [gamma, 0.0, 0.0],
                [(1.0-gamma)/2.0, gamma, 0.0],
                [1.0/(4.0*gamma), 1.0-1.0/(4.0*gamma), gamma]
            ])
            self.c = np.array([gamma, (1.0+gamma)/2.0, 1.0])
            self.b = np.array([1.0/(4.0*gamma), 1.0-1.0/(4.0*gamma), gamma])
            self.order = 3
            self.embedded_order = None
            self.bhat = None
            
        elif method == "ESDIRK3":
            # ESDIRK d'ordre 3 (Explicit first stage)
            gamma = 0.4358665215084589994160194
            self.A = np.array([
                [0.0, 0.0, 0.0],
                [0.87173304301691, gamma, 0.0],
                [0.84457060015369, -0.12990812375553, gamma]
            ])
            self.c = np.array([0.0, 0.87173304301691 + gamma, 0.84457060015369 - 0.12990812375553 + gamma])
            self.b = np.array([0.84457060015369, -0.12990812375553, gamma])
            self.order = 3
            self.embedded_order = None
            self.bhat = None
            
        elif method == "ESDIRK4":
            # ESDIRK d'ordre 4 (méthode L-stable)
            gamma = 0.25
            self.A = np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.25, 0.0, 0.0, 0.0],
                [0.17, 0.5-gamma-0.17, gamma, 0.0, 0.0],
                [0.39, 0.25-0.39, 0.45-gamma, gamma, 0.0],
                [0.15, 0.2-0.15, 0.6-gamma-0.2, 0.25, gamma]
            ])
            self.c = np.array([0.0, 0.75, 0.5, 0.75, 1.0])
            self.b = np.array([0.15, 0.2-0.15, 0.6-gamma-0.2, 0.25, gamma])
            self.order = 4
            self.embedded_order = None
            self.bhat = None
            
        # Méthodes optimisées d'ordre 3 tirées de l'article
        elif method == "ESDIRK3(2)4L[2]SA":
            # Méthode d'ordre 3 (4 étapes) avec estimateur d'ordre 2 
            # D'après le tableau A.1 de l'article
            gamma = 0.4358665215084589994160194
            
            # Coefficients de la matrice A (Tableau de la page 229, début de section 7.1)
            self.A = np.zeros((4, 4))
            self.A[0, 0] = 0.0
            self.A[1, 0] = 2*gamma
            self.A[1, 1] = gamma
            self.A[2, 0] = 0.3212788860629806
            self.A[2, 1] = -0.0590653190764472
            self.A[2, 2] = gamma
            self.A[3, 0] = 0.1932192650401376
            self.A[3, 1] = -0.8541019654728148
            self.A[3, 2] = 1.2249162788842579
            self.A[3, 3] = gamma
            
            # Calcul des abscisses c
            self.c = np.zeros(4)
            for i in range(4):
                self.c[i] = np.sum(self.A[i, :])
            
            # b = dernière ligne de A (stiffly-accurate)
            self.b = self.A[3, :]
            
            # Coefficients pour l'estimateur d'erreur embedded (ordre 2) - Eq. 28
            self.bhat = np.array([0.2000000000000000, -0.5000000000000000, 0.2179034442100000, 0.0820965557900000])
            
            self.order = 3
            self.embedded_order = 2
            
        elif method == "ESDIRK3(2)5L[2]SA":
            # Méthode d'ordre 3 (5 étapes) avec gamma = 9/40 = 0.225 
            # Coefficients d'après la table 6 et section 7.2 de l'article
            gamma = 9.0/40.0
            
            # Coefficients de la matrice A
            self.A = np.zeros((5, 5))
            # Première ligne (étape explicite)
            self.A[0, 0] = 0.0
            # Deuxième ligne
            self.A[1, 0] = 2*gamma
            self.A[1, 1] = gamma
            # Troisième ligne - avec c3 = 9(2+sqrt(2))/40
            c3 = 9.0*(2.0 + np.sqrt(2.0))/40.0
            self.A[2, 0] = (c3 - gamma)/2.0
            self.A[2, 1] = (c3 - gamma)/2.0
            self.A[2, 2] = gamma
            # Quatrième ligne - avec c4 = 3/5
            c4 = 3.0/5.0
            self.A[3, 0] = 0.0
            self.A[3, 1] = (c4 - gamma - self.A[3, 0])/2.0
            self.A[3, 2] = (c4 - gamma - self.A[3, 0])/2.0
            self.A[3, 3] = gamma
            # Cinquième ligne (dernière)
            self.A[4, 0] = 0.0
            self.A[4, 3] = (1.0 - gamma - self.A[4, 0])/3.0
            self.A[4, 1] = (1.0 - gamma - self.A[4, 0] - self.A[4, 3])/2.0
            self.A[4, 2] = (1.0 - gamma - self.A[4, 0] - self.A[4, 1] - self.A[4, 3])
            self.A[4, 4] = gamma
            
            # Calcul des abscisses c
            self.c = np.zeros(5)
            for i in range(5):
                self.c[i] = np.sum(self.A[i, :])
            
            # Coefficients b = dernière ligne de A (stiffly-accurate)
            self.b = self.A[4, :]
            
            # Coefficients embarqués d'ordre 2
            self.bhat = np.array([0.0, 0.25, 0.25, 0.25, 0.25])
            
            self.order = 3
            self.embedded_order = 2
            
        # Méthodes optimisées d'ordre 4 tirées de l'article
        elif method == "ESDIRK4(3)6L[2]SA1":
            # Méthode d'ordre 4 (6 étages) avec gamma = 1/4
            # D'après le tableau A.1 de l'article
            gamma = 0.25
            
            self.A = np.zeros((6, 6))
            # Première ligne (étape explicite)
            self.A[0, 0] = 0.0
            # Deuxième ligne
            self.A[1, 0] = 2*gamma
            self.A[1, 1] = gamma
            # Troisième ligne
            c3 = 8.0/15.0
            a31 = (c3 - 2*gamma)/6.0
            a32 = (c3 - 2*gamma)/6.0
            self.A[2, 0] = a31
            self.A[2, 1] = a32
            self.A[2, 2] = gamma
            # Quatrième ligne
            c4 = 2.0/3.0
            a41 = 8611.0/62500.0
            a42 = -1743.0/31250.0
            a43 = (c4 - a41 - a42 - gamma)
            self.A[3, 0] = a41
            self.A[3, 1] = a42
            self.A[3, 2] = a43
            self.A[3, 3] = gamma
            # Cinquième ligne
            c5 = 0.9
            a51 = 5012029.0/34652500.0
            a52 = -654441.0/2922500.0
            a53 = 174375.0/388108.0
            a54 = (c5 - a51 - a52 - a53 - gamma)
            self.A[4, 0] = a51
            self.A[4, 1] = a52
            self.A[4, 2] = a53
            self.A[4, 3] = a54
            self.A[4, 4] = gamma
            # Sixième ligne (coefficients b) - d'après le tableau 9 de l'article
            self.A[5, 0] = 115062.0/2845989.0
            self.A[5, 1] = 0.0
            self.A[5, 2] = 1097883.0/4675329.0
            self.A[5, 3] = 3867923.0/9096056.0
            self.A[5, 4] = 2575211.0/16711741.0
            self.A[5, 5] = gamma
            
            # Calcul des abscisses c
            self.c = np.zeros(6)
            for i in range(6):
                self.c[i] = np.sum(self.A[i, :])
                
            # Coefficients b = dernière ligne de A (stiffly-accurate)
            self.b = self.A[5, :]
            
            # Coefficients embarqués d'ordre 3
            self.bhat = np.array(
                [21.0/200.0, 0.0, 1097883.0/4675329.0, 3867923.0/9096056.0, 2575211.0/16711741.0, 0.0]
            )
            
            self.order = 4
            self.embedded_order = 3
            
        elif method == "ESDIRK4(3)6L[2]SA2":
            # Méthode d'ordre 4 avec gamma = 0.248
            # D'après le tableau 9 de l'annexe A de l'article
            gamma = 248.0/1000.0
            
            self.A = np.zeros((6, 6))
            # Première ligne (étape explicite)
            self.A[0, 0] = 0.0
            # Deuxième ligne
            self.A[1, 0] = 2*gamma
            self.A[1, 1] = gamma
            # Troisième ligne - d'après tableau 9 de l'annexe A
            self.A[2, 0] = -3602865186177.0/14585480527.0
            self.A[2, 1] = 31.0/125.0
            self.A[2, 2] = gamma
            # Quatrième ligne
            self.A[3, 0] = -5063886934975.0/37754990171.0
            self.A[3, 1] = 7149918333491.0/13390931526268.0
            self.A[3, 2] = 31.0/125.0
            self.A[3, 3] = gamma
            # Cinquième ligne
            self.A[4, 0] = -762830543893.0/11061539393788.0
            self.A[4, 1] = 21592626537567.0/14352247503901.0
            self.A[4, 2] = 11630056083252.0/17263101053231.0
            self.A[4, 3] = 31.0/125.0
            self.A[4, 4] = gamma
            # Sixième ligne (dernière)
            self.A[5, 0] = -12917657251.0/5222094901039.0
            self.A[5, 1] = 5602338284630.0/15643096342197.0
            self.A[5, 2] = 9002339615474.0/18125249312447.0
            self.A[5, 3] = -2420307481369.0/24731958684496.0
            self.A[5, 4] = 31.0/125.0
            self.A[5, 5] = gamma
            
            # Calcul des abscisses c
            self.c = np.zeros(6)
            for i in range(6):
                self.c[i] = np.sum(self.A[i, :])
                
            # Coefficients b = dernière ligne de A (stiffly-accurate)
            self.b = self.A[5, :]
            
            # Coefficients embarqués d'ordre 3
            self.bhat = np.array([
                -1007911106287.0/12117826057527.0,
                17694008993113.0/35931961998873.0,
                5816803040497.0/11256217655929.0,
                -538664890905.0/7490061179786.0,
                2032560730450.0/8872919773257.0,
                gamma
            ])
            
            self.order = 4
            self.embedded_order = 3
            
        elif method == "ESDIRK4(3)7L[2]SA":
            # Méthode d'ordre 4 à 7 étages avec gamma = 1/8
            # D'après tableau 10 de l'annexe A
            gamma = 1.0/8.0
            
            self.A = np.zeros((7, 7))
            # Première ligne (étape explicite)
            self.A[0, 0] = 0.0
            # Deuxième ligne
            self.A[1, 0] = 2*gamma
            self.A[1, 1] = gamma
            # Troisième ligne
            self.A[2, 0] = -391883478781.0/13744654945.0
            self.A[2, 1] = 1.0/8.0
            self.A[2, 2] = gamma
            # Quatrième ligne
            self.A[3, 0] = 17488747422135.0/168247530883.0
            self.A[3, 1] = -1748874742213.0/5795261096931.0
            self.A[3, 2] = 1.0/8.0
            self.A[3, 3] = gamma
            # Cinquième ligne
            self.A[4, 0] = -642934099309.0/717896796106705.0
            self.A[4, 1] = 9711656375562.0/10370074603625.0
            self.A[4, 2] = 1137589605079.0/3216875020685.0
            self.A[4, 3] = 1.0/8.0
            self.A[4, 4] = gamma
            # Sixième ligne
            self.A[5, 0] = 4051696060991.0/734380148729.0
            self.A[5, 1] = -264468840649.0/6105657584947.0
            self.A[5, 2] = 118647369377.0/6233854714037.0
            self.A[5, 3] = 683008737625.0/4934655825458.0
            self.A[5, 4] = 1.0/8.0
            self.A[5, 5] = gamma
            # Septième ligne (coefficients b)
            self.A[6, 0] = -5649241495537.0/14093099002237.0
            self.A[6, 1] = 5718691255176.0/6089204655961.0
            self.A[6, 2] = 2199600963556.0/4241893152925.0
            self.A[6, 3] = 8860614275765.0/11425531467341.0
            self.A[6, 4] = -3696041814078.0/6641566663007.0
            self.A[6, 5] = 1.0/8.0
            self.A[6, 6] = gamma
            
            # Calcul des abscisses c
            self.c = np.zeros(7)
            for i in range(7):
                self.c[i] = np.sum(self.A[i, :])
                
            # Coefficients b = dernière ligne de A (stiffly-accurate)
            self.b = self.A[6, :]
            
            # Coefficients embarqués d'ordre 3
            self.bhat = np.array([
                -1517409284625.0/6267517876163.0,
                8291371032348.0/12587291883523.0,
                5328310281212.0/10646448185159.0,
                5405006853541.0/7104492075037.0,
                -4254786582061.0/7445269677723.0,
                19.0/140.0,
                gamma
            ])
            
            self.order = 4
            self.embedded_order = 3
            
        # Méthodes d'ordre 5 tirées de l'article
        elif method == "ESDIRK5(4)7L[2]SA1":
            # Méthode d'ordre 5 à 7 étages avec gamma = 23/125 = 0.184
            # D'après le tableau 11 de l'article
            gamma = 23.0/125.0
            
            # Coefficients extraits du tableau 11 de l'annexe A
            self.A = np.zeros((7, 7))
            # Première ligne (étape explicite)
            self.A[0, 0] = 0.0
            # Deuxième ligne
            self.A[1, 0] = 2*gamma
            self.A[1, 1] = gamma
            # Troisième ligne
            self.A[2, 0] = 7910200473043.0/561426431547.0
            self.A[2, 1] = 23.0/125.0
            self.A[2, 2] = gamma
            # Quatrième ligne
            self.A[3, 0] = -15815907635811.0/257294102345.0
            self.A[3, 1] = -85517644447.0/5003708988389.0
            self.A[3, 2] = 23.0/125.0
            self.A[3, 3] = gamma
            # Cinquième ligne
            self.A[4, 0] = -16533271115804.0/48416487981.0
            self.A[4, 1] = 1514767744496.0/9099671765375.0
            self.A[4, 2] = 14283835447591.0/12247432691556.0
            self.A[4, 3] = 23.0/125.0
            self.A[4, 4] = gamma
            # Sixième ligne
            self.A[5, 0] = -45400119708258.0/418487046959.0
            self.A[5, 1] = -1790937573418.0/7393406387169.0
            self.A[5, 2] = 10819093665085.0/7266595846747.0
            self.A[5, 3] = 4109463131231.0/7386972500302.0
            self.A[5, 4] = 23.0/125.0
            self.A[5, 5] = gamma
            # Septième ligne (coefficients b)
            self.A[6, 0] = -188593204321.0/4778616380481.0
            self.A[6, 1] = 2809310203510.0/10304234040467.0
            self.A[6, 2] = 1021729336898.0/2364210264653.0
            self.A[6, 3] = 870612361811.0/2470410392208.0
            self.A[6, 4] = -1307970675534.0/8059683598661.0
            self.A[6, 5] = 23.0/125.0
            self.A[6, 6] = gamma
            
            # Calcul des abscisses c
            self.c = np.zeros(7)
            for i in range(7):
                self.c[i] = np.sum(self.A[i, :])
                
            # Coefficients b = dernière ligne de A (stiffly-accurate)
            self.b = self.A[6, :]
            
            # Coefficients embarqués d'ordre 4
            self.bhat = np.array([
                -582099335757.0/7214068459310.0,
                615023338567.0/3362626566945.0,
                3192122436311.0/6174152374399.0,
                6156034052041.0/14430468657929.0,
                -1011318518279.0/9693750372484.0,
                1914490192573.0/13754262428401.0,
                gamma
            ])
            
            self.order = 5
            self.embedded_order = 4
            
        elif method == "ESDIRK5(4)7L[2]SA2":
            # Méthode d'ordre 5 à 7 étages avec gamma = 23/125 (variante)
            # D'après le tableau 11 de l'annexe A
            gamma = 23.0/125.0
            
            self.A = np.zeros((7, 7))
            # Première ligne (étape explicite)
            self.A[0, 0] = 0.0
            # Deuxième ligne
            self.A[1, 0] = 2*gamma
            self.A[1, 1] = gamma
            # Troisième ligne (avec c3 = 7121331996143/11335814405378)
            self.A[2, 0] = 7910200473043.0/561426431547.0
            self.A[2, 1] = 23.0/125.0
            self.A[2, 2] = gamma
            # Quatrième ligne (avec c4 = 49/353)
            self.A[3, 0] = -15815907635811.0/257294102345.0
            self.A[3, 1] = -85517644447.0/5003708988389.0
            self.A[3, 2] = 23.0/125.0
            self.A[3, 3] = gamma
            # Cinquième ligne (avec c5 = 3706679970760/5295570149437)
            self.A[4, 0] = -16533271115804.0/48416487981.0
            self.A[4, 1] = 1514767744496.0/9099671765375.0
            self.A[4, 2] = 14283835447591.0/12247432691556.0
            self.A[4, 3] = 23.0/125.0
            self.A[4, 4] = gamma
            # Sixième ligne (avec c6 = 347/382)
            self.A[5, 0] = -45400119708258.0/418487046959.0
            self.A[5, 1] = -1790937573418.0/7393406387169.0
            self.A[5, 2] = 10819093665085.0/7266595846747.0
            self.A[5, 3] = 4109463131231.0/7386972500302.0
            self.A[5, 4] = 23.0/125.0
            self.A[5, 5] = gamma
            # Septième ligne (dernière)
            self.A[6, 0] = -188593204321.0/4778616380481.0
            self.A[6, 1] = 2809310203510.0/10304234040467.0
            self.A[6, 2] = 1021729336898.0/2364210264653.0
            self.A[6, 3] = 870612361811.0/2470410392208.0
            self.A[6, 4] = -1307970675534.0/8059683598661.0
            self.A[6, 5] = 23.0/125.0
            self.A[6, 6] = gamma
            
            # Calcul des abscisses c
            self.c = np.zeros(7)
            for i in range(7):
                self.c[i] = np.sum(self.A[i, :])
                
            # Coefficients b = dernière ligne de A (stiffly-accurate)
            self.b = self.A[6, :]
            
            # Coefficients embarqués d'ordre 4
            self.bhat = np.array([
                -582099335757.0/7214068459310.0,
                615023338567.0/3362626566945.0,
                3192122436311.0/6174152374399.0,
                6156034052041.0/14430468657929.0,
                -1011318518279.0/9693750372484.0,
                1914490192573.0/13754262428401.0,
                gamma
            ])
            
            self.order = 5
            self.embedded_order = 4
            
        elif method == "ESDIRK5(4)8L[2]SA":
            # Méthode d'ordre 5 à 8 étages avec gamma = 1/7
            # D'après tableau 12 de l'annexe A
            gamma = 1.0/7.0
            
            self.A = np.zeros((8, 8))
            # Première ligne (étape explicite)
            self.A[0, 0] = 0.0
            # Deuxième ligne
            self.A[1, 0] = 2*gamma
            self.A[1, 1] = gamma
            # Troisième ligne
            self.A[2, 0] = 1521428834970.0/8822750406821.0
            self.A[2, 1] = 1.0/7.0
            self.A[2, 2] = gamma
            # Quatrième ligne
            self.A[3, 0] = 5338711108027.0/29869763600956.0
            self.A[3, 1] = 1483184435021.0/6216373359362.0
            self.A[3, 2] = 1.0/7.0
            self.A[3, 3] = gamma
            # Cinquième ligne
            self.A[4, 0] = 2264935805846.0/12599242299355.0
            self.A[4, 1] = 1330937762090.0/13140498839569.0
            self.A[4, 2] = -287786842865.0/17211061626069.0
            self.A[4, 3] = 1.0/7.0
            self.A[4, 4] = gamma
            # Sixième ligne
            self.A[5, 0] = 1183529370805.0/27276862197.0
            self.A[5, 1] = -2960446233093.0/7419588050389.0
            self.A[5, 2] = -3064256220847.0/46575910191280.0
            self.A[5, 3] = 6010467311487.0/7886573591137.0
            self.A[5, 4] = 1.0/7.0
            self.A[5, 5] = gamma
            # Septième ligne
            self.A[6, 0] = 11342701839199.0/703695183946.0
            self.A[6, 1] = 4862384331311.0/10104465681802.0
            self.A[6, 2] = 1127469817207.0/2459314315538.0
            self.A[6, 3] = -9518066423555.0/11243131997224.0
            self.A[6, 4] = -811155580665.0/7490894181109.0
            self.A[6, 5] = 1.0/7.0
            self.A[6, 6] = gamma
            # Huitième ligne (coefficients b)
            self.A[7, 0] = 2162042939093.0/22873479087181.0
            self.A[7, 1] = -4222515349147.0/9397994281350.0
            self.A[7, 2] = 3431955516634.0/4748630552535.0
            self.A[7, 3] = -374165068070.0/9085231819471.0
            self.A[7, 4] = -1847934966618.0/8254951855109.0
            self.A[7, 5] = 5186241678079.0/7861334770480.0
            self.A[7, 6] = 1.0/7.0
            self.A[7, 7] = gamma
            
            # Calcul des abscisses c
            self.c = np.zeros(8)
            for i in range(8):
                self.c[i] = np.sum(self.A[i, :])
                
            # Coefficients b = dernière ligne de A (stiffly-accurate)
            self.b = self.A[7, :]
            
            # Coefficients embarqués d'ordre 4
            self.bhat = np.array([
                701879993119.0/7084679725724.0,
                -8461269287478.0/14654112271769.0,
                6612459227430.0/11388259134383.0,
                2632441606103.0/12598871370240.0,
                -2147694411931.0/10286892713802.0,
                4103061625716.0/6371697724583.0,
                4103061625716.0/6371697724583.0,
                gamma
            ])
            
            self.order = 5
            self.embedded_order = 4
            
        # Méthode d'ordre 6 tirée de l'article
        elif method == "ESDIRK6(5)9L[2]SA":
            # Méthode d'ordre 6 à 9 étages avec gamma = 2/9
            # D'après tableau 13 de l'annexe A
            gamma = 2.0/9.0
            
            self.A = np.zeros((9, 9))
            # Première ligne (étape explicite)
            self.A[0, 0] = 0.0
            # Deuxième ligne
            self.A[1, 0] = 2*gamma
            self.A[1, 1] = gamma
            # Troisième ligne
            self.A[2, 0] = 1.0/9.0
            self.A[2, 1] = -52295652026801.0/1014133226193379.0
            self.A[2, 2] = gamma
            # Quatrième ligne
            self.A[3, 0] = 37633260247889.0/456511413219805.0
            self.A[3, 1] = -162541608159785.0/642690962402252.0
            self.A[3, 2] = 186915148640310.0/408032288622937.0
            self.A[3, 3] = gamma
            # Cinquième ligne
            self.A[4, 0] = -37161579357179.0/532208945751958.0
            self.A[4, 1] = -211140841282847.0/266150973773621.0
            self.A[4, 2] = 884359688045285.0/894827558443789.0
            self.A[4, 3] = 845261567597837.0/1489150009616527.0
            self.A[4, 4] = gamma
            # Sixième ligne
            self.A[5, 0] = 32386175866773.0/281337331200713.0
            self.A[5, 1] = 498042629717897.0/1553069719539220.0
            self.A[5, 2] = -73718535152787.0/262520491717733.0
            self.A[5, 3] = -147656452213061.0/931530156064788.0
            self.A[5, 4] = -16605385309793.0/2106054502776008.0
            self.A[5, 5] = gamma
            # Septième ligne
            self.A[6, 0] = -38317091100349.0/1495803980405525.0
            self.A[6, 1] = 233542892858682.0/880478953581929.0
            self.A[6, 2] = -281992829959331.0/709729395317651.0
            self.A[6, 3] = -52133614094227.0/895217507304839.0
            self.A[6, 4] = -9321507955616.0/673810579175161.0
            self.A[6, 5] = 79481371174259.0/817241804646218.0
            self.A[6, 6] = gamma
            # Huitième ligne
            self.A[7, 0] = -486324380411713.0/1453057025607868.0
            self.A[7, 1] = -1085539098090580.0/1176943702490991.0
            self.A[7, 2] = 370161554881539.0/461122320759884.0
            self.A[7, 3] = 804017943088158.0/886363045286999.0
            self.A[7, 4] = -15204170533868.0/934878849212545.0
            self.A[7, 5] = -248215443403879.0/815097869999138.0
            self.A[7, 6] = 339987959782520.0/552150039467091.0
            self.A[7, 7] = gamma
            # Neuvième ligne (coefficients b)
            self.A[8, 0] = 0.0
            self.A[8, 1] = 0.0
            self.A[8, 2] = 281246836687281.0/672805784366875.0
            self.A[8, 3] = 250674029546725.0/464056298040646.0
            self.A[8, 4] = 88917245119922.0/798581755375683.0
            self.A[8, 5] = 127306093275639.0/658941305589808.0
            self.A[8, 6] = -319515475352107.0/658842144391777.0
            self.A[8, 7] = gamma
            self.A[8, 8] = gamma
            
            # Calcul des abscisses c
            self.c = np.zeros(9)
            for i in range(9):
                self.c[i] = np.sum(self.A[i, :])
                
            # Coefficients b = dernière ligne de A (stiffly-accurate)
            self.b = self.A[8, :]
            
            # Coefficients embarqués d'ordre 5
            self.bhat = np.array([
                -204006714482445.0/253120897457864.0,
                0.0,
                -8180624343107.0/19743038324217.0,
                3176520686137389.0/1064235527052079.0,
                -574817982095666.0/1374329821545869.0,
                -507643245828272.0/1001056758847831.0,
                2013538191006793.0/972919262949000.0,
                352681731710820.0/726444701718347.0,
                -12107714797721.0/746708658438760.0
            ])
            
            self.order = 6
            self.embedded_order = 5
            
        else:
            raise ValueError(f"Méthode DIRK inconnue: {method}")
        
        self.num_stages = len(self.b)
        self.method = method
    
    def __str__(self):
        """Représentation sous forme de chaîne de caractères avec informations détaillées"""
        info = f"DIRK method: {self.method}\n"
        info += f"Order: {self.order}\n"
        if self.embedded_order:
            info += f"Embedded order: {self.embedded_order}\n"
        info += f"Stages: {self.num_stages}\n"
        info += f"Diagonal coefficient γ: {self.A[1, 1]:.6f}\n"
        info += "L-stability: Yes (optimal for stiff problems)\n"
        info += "Stage-order: 2 (reduced order reduction)\n"
        info += "Stiffly-accurate: Yes (better for DAEs)\n"
        return info