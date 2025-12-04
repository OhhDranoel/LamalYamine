<?php
// Conexão
$conexao = mysqli_connect("localhost", "root", "", "login");

if (!$conexao) {
    header('Content-Type: application/json; charset=utf-8');
    die(json_encode(['sucesso' => false, 'mensagem' => 'Erro ao conectar ao banco: ' . mysqli_connect_error()]));
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
    header('Content-Type: application/json; charset=utf-8');
    echo json_encode(['sucesso' => false, 'mensagem' => 'As senhas não coincidem.']);
    exit;
}

// INSERT com 6 colunas
$sql = "INSERT INTO cadastro (nome, idade, nome_usuario, email, senha, conf_senha)
        VALUES ('$nome', '$idade', '$usuario', '$email', '$senha', '$conf_senha')";

$resultado = mysqli_query($conexao, $sql);

// Resposta em JSON
header('Content-Type: application/json; charset=utf-8');
if ($resultado) {
    echo json_encode([
        'sucesso' => true,
        'mensagem' => 'Usuário cadastrado com sucesso!',
        'usuario' => $usuario,
        'email' => $email
    ]);
} else {
    echo json_encode(['sucesso' => false, 'mensagem' => 'Erro ao cadastrar: ' . mysqli_error($conexao)]);
}

mysqli_close($conexao);
?>
