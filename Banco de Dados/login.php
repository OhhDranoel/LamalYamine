<?php
// Conexão
$conexao = mysqli_connect("localhost", "root", "", "login");

if (!$conexao) {
    die("Erro ao conectar ao banco: " . mysqli_connect_error());
}

// Dados do formulário
$usuario = $_POST['nome_usuario'];
$senha = $_POST['senha'];

// INSERT NA NOVA TABELA
$sql = "INSERT INTO criar_conta (nome_usuario, senha) VALUES ('$usuario', '$senha')";
$resultado = mysqli_query($conexao, $sql);

// Resposta
if ($resultado) {
    echo "Usuário cadastrado com sucesso!";
} else {
    echo "Erro ao cadastrar: " . mysqli_error($conexao);
}
?>
