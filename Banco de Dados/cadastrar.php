<?php
// Conexão
$conexao = mysqli_connect("localhost", "root", "", "login");

if (!$conexao) {
    die("Erro ao conectar ao banco: " . mysqli_connect_error());
}

// Dados do formulário
$nome = $_POST['nome'];
$idade = $_POST['idade'];
$usuario = $_POST['nome_usuario'];
$email = $_POST['email'];
$senha = $_POST['senha'];
$conf_senha = $_POST['conf_senha'];

// Verificar se as senhas coincidem
if ($senha !== $conf_senha) {
    die("Erro: As senhas não coincidem.");
}

// INSERT com 6 colunas
$sql = "INSERT INTO cadastro (nome, idade, nome_usuario, email, senha, conf_senha)
        VALUES ('$nome', '$idade', '$usuario', '$email', '$senha', '$conf_senha')";

$resultado = mysqli_query($conexao, $sql);

// Resposta
if ($resultado) {
    echo "Usuário cadastrado com sucesso!";
} else {
    echo "Erro ao cadastrar: " . mysqli_error($conexao);
}
?>
