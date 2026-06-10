import tkinter as tk
from tkinter import ttk, messagebox
import data
import modelo_reg_log
import modelo_arvore

CSV_PATH = "./data/titanic.csv"

INFO = {
    "name": (
        "Nome do passageiro",
        "Este campo não afeta a predição.\n\n"
        "Mas é legal :p"
    ),
    "pclass": (
        "Classe do passageiro",
        "1: Primeira Classe - Cabines nos conveses superiores (meia-nau), próximo aos botes salva-vidas.\n\n"
        "2: Segunda Classe - Cabines na popa do navio. \n\n"
        "3: Terceira Classe - Cabines nas extremidades inferiores do navio (popa e proa), separados por grades."
    ),
    "age": (
        "Idade",
        "Crianças e idosos foram evacuadas primeiro.\n\n"
        "Valores vazios Missing ages in the dataset are filled with the median (~28 years)."
    ),
    "fam": (
        "Família a bordo",
        "Soma de irmãos, conjugês, pais e filhos embarcados.\n\n"
    ),
    "sex": (
        "Sexo",
        "'Mulheres e crianças primeiro!' foi a regra principal no momento da evacuação.\n\n"
    ),
    "embarked": (
        "Porto de embarque",
        "Southampton (S) — Porto de partida original. Concentrou a maior parte dos embarques e recebeu passageiros de todas as classes.\n\n"
        "Cherbourg (C) —  Primeira parada da viagem. Destacou-se pelo embarque de passageiros ricos de elite e figuras ilustres da primeira classe.\n\n"
        "Queenstown (Q) — Última parada antes do Oceano Atlântico. A grande maioria dos que embarcaram ali eram imigrantes da terceira classe."
    ),
    "model_select": (
        "Modelo de Machine Learning",
        "Escolha o algoritmo usado para prever a sobrevivência.\n\n"
        "Regressão Logística: Estima probabilidades usando uma curva matemática. Ótimo para relações lineares.\n"
        "Árvore de Decisão: Cria regras de 'se-então' (ex: Se idade < 10 e sexo=F). Fácil de visualizar e entender."
    ),
    "lr_C": (
        "Força da Regularização (C)",
        "Controla a penalidade por modelos muito complexos.\n\n"
        "Valores menores (ex: 0.1) criam modelos mais simples e generalistas (evita overfitting).\n\n"
        "Valores maiores (ex: 10.0) dão mais liberdade ao modelo para se ajustar aos dados de treino."
    ),
    "lr_iter": (
        "Máximo de Iterações",
        "Número máximo de tentativas que o modelo tem para encontrar os melhores pesos matemáticos.\n\n"
        "Aumente este valor se o modelo avisar que falhou em convergir."
    ),
    "lr_solver": (
        "Solver (Otimizador)",
        "O algoritmo matemático usado para encontrar a solução.\n\n"
        "'lbfgs': Padrão, rápido e bom para maioria dos casos.\n"
        "'saga': Bom para datasets muito grandes.\n"
        "'liblinear': Bom para datasets pequenos."
    ),
    "max_depth": (
        "Profundidade Máxima (max_depth)",
        "O tamanho máximo da árvore de decisão.\n\n"
        "Árvores muito fundas (ex: 10) podem decorar o dataset (overfitting). Árvores rasas (ex: 3) são mais generalistas."
    ),
    "min_leaf": (
        "Mínimo de Amostras na Folha",
        "Quantos passageiros, no mínimo, devem estar no fim de um ramo para ele ser criado.\n\n"
        "Valores maiores evitam que o modelo crie regras hiperespecíficas para uma única pessoa."
    ),
    "dt_crit": (
        "Critério de Divisão",
        "Como a árvore decide a melhor pergunta a fazer (ex: Dividir por sexo ou por idade?).\n\n"
        "'gini': Mede a impureza do grupo.\n"
        "'entropy': Mede o ganho de informação. \n\nAmbos costumam dar resultados parecidos."
    ),
    "random_state": (
        "Semente Aleatória (random_state)",
        "Garante que os resultados sejam reproduzíveis.\n\n"
        "Algoritmos de ML usam aleatoriedade interna (ex: para desempatar regras ou inicializar pesos). "
        "Fixar este número faz com que o modelo dê sempre o mesmo resultado matemático se os dados forem os mesmos."
    ),
    "random_state": (
        "Semente Aleatória (random_state)",
        "Garante que os resultados sejam reproduzíveis.\n\n"
        "Fixar este número faz com que o modelo dê sempre o mesmo resultado matemático se os dados forem os mesmos."
    )
    
}

PORT_IMAGES = {
    "S": "./images/southampton.png",
    "C": "./images/cherbourg.png",
    "Q": "./images/queenstown.png",
}

CLASS_IMAGES = {
    1: "./images/titanic_class1.png",
    2: "./images/titanic_class2.png",
    3: "./images/titanic_class3.png",
}


def load_image(path, max_w=600, max_h=400):
    """Load a local image and return a Tkinter PhotoImage scaled to fit."""
    try:
        from PIL import Image, ImageTk
        img = Image.open(path)
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception:
        return None


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Titanic previsões.")
        self.resizable(False, False)
        self.minsize(600, 300)

        try:
            self.X, self.X_scaled, self.y, self.scaler = data.load_and_prepare(CSV_PATH)
        except Exception as e:
            messagebox.showerror("Erro no datase",
                                 f"Falha ao importar o dataset em {CSV_PATH}\n\n{e}")
            self.destroy()
            return

        self._img_cache = {}

        self._build()

    def _build(self):
        left  = tk.Frame(self, padx=14, pady=14)
        left.grid(row=0, column=0, sticky="ns")

        ttk.Separator(self, orient="vertical").grid(
            row=0, column=1, sticky="ns", padx=4)

        right = tk.Frame(self, padx=14, pady=14, width=400)
        right.grid(row=0, column=2, sticky="nsew")
        right.grid_propagate(False)

        self._build_form(left)
        self._build_info(right)

    def _build_form(self, p):
        pad = dict(padx=6, pady=4)

        def label(text, row):
            tk.Label(p, text=text).grid(row=row, column=0, sticky="w", **pad)

        label("None", 0)
        self.name_var = tk.StringVar()
        e = tk.Entry(p, textvariable=self.name_var, width=22)
        e.grid(row=0, column=1, sticky="w", **pad)
        e.bind("<FocusIn>", lambda _: self._show_info("name"))

        # Class
        label("Classe da cabine", 1)
        self.pclass_var = tk.IntVar(value=3)
        cb_class = ttk.Combobox(p, textvariable=self.pclass_var,
                                values=[1, 2, 3], state="readonly", width=6)
        cb_class.grid(row=1, column=1, sticky="w", **pad)
        cb_class.bind("<<ComboboxSelected>>",
                      lambda _: self._show_info("pclass", image_key=self.pclass_var.get(),
                                                image_map=CLASS_IMAGES))
        cb_class.bind("<FocusIn>",
                      lambda _: self._show_info("pclass", image_key=self.pclass_var.get(),
                                                image_map=CLASS_IMAGES))

        label("Idade", 2)
        self.age_var = tk.IntVar(value=30)
        sb_age = tk.Spinbox(p, textvariable=self.age_var, from_=1, to=80, width=6)
        sb_age.grid(row=2, column=1, sticky="w", **pad)
        sb_age.bind("<FocusIn>", lambda _: self._show_info("age"))
        
        label("Família a bordo", 3)
        self.fam_var = tk.IntVar(value=0)
        sb_fam = tk.Spinbox(p, textvariable=self.fam_var, from_=0, to=10, width=6)
        sb_fam.grid(row=3, column=1, sticky="w", **pad)
        sb_fam.bind("<FocusIn>", lambda _: self._show_info("fam"))

        label("Sex", 4)
        self.sex_var = tk.StringVar(value="Male")
        cb_sex = ttk.Combobox(p, textvariable=self.sex_var,
                              values=["Male", "Female"], state="readonly", width=10)
        cb_sex.grid(row=4, column=1, sticky="w", **pad)
        cb_sex.bind("<<ComboboxSelected>>", lambda _: self._show_info("sex"))
        cb_sex.bind("<FocusIn>",            lambda _: self._show_info("sex"))

        label("Porto de embarque", 5)
        self.embarked_var = tk.StringVar(value="S — Southampton")
        cb_port = ttk.Combobox(p, textvariable=self.embarked_var,
                               values=["S — Southampton",
                                       "C — Cherbourg",
                                       "Q — Queenstown"],
                               state="readonly", width=18)
        cb_port.grid(row=5, column=1, sticky="w", **pad)
        cb_port.bind("<<ComboboxSelected>>",
                     lambda _: self._show_info("embarked",
                                               image_key=self.embarked_var.get()[0],
                                               image_map=PORT_IMAGES))
        cb_port.bind("<FocusIn>",
                     lambda _: self._show_info("embarked",
                                               image_key=self.embarked_var.get()[0],
                                               image_map=PORT_IMAGES))

        ttk.Separator(p, orient="horizontal").grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=8)

        tk.Label(p, text="Modelo").grid(row=7, column=0, sticky="w", **pad)
        self.model_var = tk.StringVar(value="Regressão Logística")
        cb_model = ttk.Combobox(p, textvariable=self.model_var,
                     values=["Regressão Logística", "Árvore de Decisão"],
                     state="readonly", width=18)
        cb_model.grid(row=7, column=1, sticky="w", **pad)
        cb_model.bind("<FocusIn>", lambda _: self._show_info("model_select"))
        cb_model.bind("<<ComboboxSelected>>", lambda _: self._show_info("model_select"))
        self.model_var.trace_add("write", lambda *_: self._swap_params())

        self.params_frame = tk.Frame(p)
        self.params_frame.grid(row=8, column=0, columnspan=2, sticky="w")
        
        self._swap_params()

        ttk.Separator(p, orient="horizontal").grid(
            row=9, column=0, columnspan=2, sticky="ew", pady=8)

        tk.Button(p, text="Treinar e Prever", command=self._predict,
                  width=18).grid(row=10, column=0, columnspan=2, pady=4)

        self.result_var = tk.StringVar()
        self.result_lbl = tk.Label(p, textvariable=self.result_var,
                                   font=("TkDefaultFont", 12, "bold"),
                                   justify="center")
        self.result_lbl.grid(row=11, column=0, columnspan=2, pady=(6, 2))

        self.detail_lbl = tk.Label(p, text="", fg="gray", justify="center")
        self.detail_lbl.grid(row=12, column=0, columnspan=2)

    def _build_info(self, p):
        self.info_title = tk.Label(p, text="", font=("TkDefaultFont", 10, "bold"),
                                   wraplength=256, justify="left", anchor="w")
        self.info_title.pack(anchor="w", pady=(0, 4))

        self.info_body = tk.Label(p, text="← Clique nos campos para saber mais",
                                  wraplength=256, justify="left", anchor="nw",
                                  fg="gray")
        self.info_body.pack(anchor="w")

        self.info_img_lbl = tk.Label(p)
        self.info_img_lbl.pack(anchor="w", pady=(10, 0))

    def _show_info(self, key, image_key=None, image_map=None):
        title, body = INFO[key]
        self.info_title.config(text=title)
        self.info_body.config(text=body, fg="black")

        self.info_img_lbl.config(image="")
        self.info_img_lbl.image = None

        if image_key is not None and image_map is not None:
            url = image_map.get(image_key)
            if url:
                cache_key = f"{image_key}"
                if cache_key not in self._img_cache:
                    self._img_cache[cache_key] = load_image(url)
                img = self._img_cache.get(cache_key)
                if img:
                    self.info_img_lbl.config(image=img)
                    self.info_img_lbl.image = img   

    def _swap_params(self):
        for w in self.params_frame.winfo_children():
            w.destroy()
        pad = dict(padx=6, pady=3)
        
        if self.model_var.get() == "Regressão Logística":
            # --- Random State (Logistic Regression) ---
            tk.Label(self.params_frame, text="random_state").grid(row=3, column=0, sticky="w", **pad)
            self.lr_rs_var = tk.IntVar(value=42)
            sb_lr_rs = tk.Spinbox(self.params_frame, textvariable=self.lr_rs_var,
                       from_=0, to=9999, width=8)
            sb_lr_rs.grid(row=3, column=1, sticky="w", **pad)
            sb_lr_rs.bind("<FocusIn>", lambda _: self._show_info("random_state"))
            
            # --- C (Regularização) ---
            tk.Label(self.params_frame, text="C (regularização)").grid(row=0, column=0, sticky="w", **pad)
            self.lr_C_var = tk.DoubleVar(value=1.0)
            sb_c = tk.Spinbox(self.params_frame, textvariable=self.lr_C_var,
                       from_=0.01, to=10.0, increment=0.1,
                       format="%.2f", width=8)
            sb_c.grid(row=0, column=1, sticky="w", **pad)
            sb_c.bind("<FocusIn>", lambda _: self._show_info("lr_C"))

            # --- Max Iterations ---
            tk.Label(self.params_frame, text="max_iterations").grid(row=1, column=0, sticky="w", **pad)
            self.lr_iter_var = tk.IntVar(value=200)
            sb_iter = tk.Spinbox(self.params_frame, textvariable=self.lr_iter_var,
                       from_=50, to=2000, increment=50,
                       width=8)
            sb_iter.grid(row=1, column=1, sticky="w", **pad)
            sb_iter.bind("<FocusIn>", lambda _: self._show_info("lr_iter"))

            # --- Solver ---
            tk.Label(self.params_frame, text="solver").grid(row=2, column=0, sticky="w", **pad)
            self.lr_solver_var = tk.StringVar(value="lbfgs")
            cb_solver = ttk.Combobox(self.params_frame, textvariable=self.lr_solver_var,
                         values=["lbfgs", "saga", "liblinear"],
                         state="readonly", width=10)
            cb_solver.grid(row=2, column=1, sticky="w", **pad)
            cb_solver.bind("<<ComboboxSelected>>", lambda _: self._show_info("lr_solver"))
            cb_solver.bind("<FocusIn>", lambda _: self._show_info("lr_solver"))
            
        else:
            # --- Random State (Decision Tree) ---
            tk.Label(self.params_frame, text="random_state").grid(row=3, column=0, sticky="w", **pad)
            self.dt_rs_var = tk.IntVar(value=42)
            sb_dt_rs = tk.Spinbox(self.params_frame, textvariable=self.dt_rs_var,
                       from_=0, to=9999, width=6)
            sb_dt_rs.grid(row=3, column=1, sticky="w", **pad)
            sb_dt_rs.bind("<FocusIn>", lambda _: self._show_info("random_state"))
            
            # --- Max Depth ---
            tk.Label(self.params_frame, text="max_depth").grid(row=0, column=0, sticky="w", **pad)
            self.dt_depth_var = tk.IntVar(value=3)
            sb_depth = tk.Spinbox(self.params_frame, textvariable=self.dt_depth_var,
                       from_=1, to=10, width=6)
            sb_depth.grid(row=0, column=1, sticky="w", **pad)
            sb_depth.bind("<FocusIn>", lambda _: self._show_info("max_depth"))

            # --- Min Samples Leaf ---
            tk.Label(self.params_frame, text="min_sample_leafs").grid(row=1, column=0, sticky="w", **pad)
            self.dt_leaf_var = tk.IntVar(value=5)
            sb_leaf = tk.Spinbox(self.params_frame, textvariable=self.dt_leaf_var,
                       from_=1, to=40, width=6)
            sb_leaf.grid(row=1, column=1, sticky="w", **pad)
            sb_leaf.bind("<FocusIn>", lambda _: self._show_info("min_leaf"))

            # --- Criterion ---
            tk.Label(self.params_frame, text="criterion").grid(row=2, column=0, sticky="w", **pad)
            self.dt_crit_var = tk.StringVar(value="gini")
            cb_crit = ttk.Combobox(self.params_frame, textvariable=self.dt_crit_var,
                         values=["gini", "entropy"],
                         state="readonly", width=10)
            cb_crit.grid(row=2, column=1, sticky="w", **pad)
            cb_crit.bind("<<ComboboxSelected>>", lambda _: self._show_info("dt_crit"))
            cb_crit.bind("<FocusIn>", lambda _: self._show_info("dt_crit"))

    def _predict(self):
        name     = self.name_var.get().strip() or "O passageiro"
        sex_male = 1 if self.sex_var.get() == "Male" else 0
        embarked = self.embarked_var.get()[0] 

        passenger = data.build_passenger(
            self.pclass_var.get(),
            self.age_var.get(),
            self.fam_var.get(),
            sex_male,
            embarked,
        )

        if self.model_var.get() == "Regressão Logística":
            p_scaled = self.scaler.transform(passenger)
            pred, proba, trained = modelo_reg_log.train_and_predict(
                self.X_scaled, self.y, p_scaled,
                C=self.lr_C_var.get(),
                max_iter=self.lr_iter_var.get(),
                solver=self.lr_solver_var.get(),
                random_state=self.lr_rs_var.get(),
            )
            acc = trained.score(self.X_scaled, self.y)
            modelo_reg_log.plot(trained, p_scaled, proba)
        else:
            pred, proba, trained = modelo_arvore.train_and_predict(
                self.X, self.y, passenger,
                max_depth=self.dt_depth_var.get(),
                criterion=self.dt_crit_var.get(),
                min_samples_leaf=self.dt_leaf_var.get(),
                random_state=self.dt_rs_var.get(), 
            )
            acc = trained.score(self.X, self.y)
            modelo_arvore.plot(trained)

        survived = bool(pred)
        if survived:
            self.result_var.set(f"{name} sobreviveu!")
            self.result_lbl.config(fg="green")
        else:
            self.result_var.set(f"{name} não sobreviveu.")
            self.result_lbl.config(fg="red")

        self.detail_lbl.config(
            text=f"Probabilidade de sobrevivência: {proba*100:.1f}%   |   Acurácia: {acc*100:.1f}%"
        )


if __name__ == "__main__":
    App().mainloop()