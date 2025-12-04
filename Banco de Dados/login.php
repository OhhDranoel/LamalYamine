<?php
// Conexão
$conexao = mysqli_connect("localhost", "root", "", "login");

if (!$conexao) {
    die("Erro ao conectar ao banco: " . mysqli_connect_error());
}

// Dados do formulário
$usuario = $_POST['nome_usuario'];
$senha = $_POST['senha'];

// Verificar se o usuário existe na tabela de cadastro
$sql = "SELECT * FROM cadastro WHERE nome_usuario = '$usuario' AND senha = '$senha'";
$resultado = mysqli_query($conexao, $sql);

// Verificar se a consulta foi bem-sucedida
if (!$resultado) {
    die("Erro na consulta: " . mysqli_error($conexao));
}

// Verificar se encontrou o usuário
if (mysqli_num_rows($resultado) > 0) {
    // Login bem-sucedido - obter dados do usuário
    $dados_usuario = mysqli_fetch_assoc($resultado);
    $email = $dados_usuario['email'];

    // Enviar header JSON e retornar os dados em formato JSON
    header('Content-Type: application/json; charset=utf-8');
    echo json_encode([
        'sucesso' => true,
        'usuario' => $usuario,
        'email' => $email
    ]);
} else {
    // Usuário não existe ou senha incorreta
    // Retornar mensagem de erro como texto (frontend trata parse fail)
    header('Content-Type: text/plain; charset=utf-8');
    echo "Usuário não existe ou senha incorreta!";
}

mysqli_close($conexao);
?>
