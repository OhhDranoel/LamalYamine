<?php
// Conexão
$conexao = mysqli_connect("localhost", "root", "", "login");

if (!$conexao) {
    die("Erro ao conectar ao banco: " . mysqli_connect_error());
}

// Receber ação
$acao = $_POST['acao'] ?? '';
$usuario = $_POST['usuario'] ?? '';

// Função auxiliar para respostas JSON
function json_response($conexao, $success, $message) {
    header('Content-Type: application/json; charset=utf-8');
    $response = array('success' => $success ? true : false, 'message' => $message);
    mysqli_close($conexao);
    echo json_encode($response);
    exit;
}

// Validar entrada
if (!$usuario) {
    json_response($conexao, false, 'Erro: Usuário não identificado');
}

switch($acao) {
    case 'alterar_senha':
        alterarSenha($conexao, $usuario);
        break;
    case 'alterar_email':
        alterarEmail($conexao, $usuario);
        break;
    case 'excluir_conta':
        excluirConta($conexao, $usuario);
        break;
    default:
        die("Ação não reconhecida");
}

function alterarSenha($conexao, $usuario) {
    $senha_atual = $_POST['senha_atual'] ?? '';
    $nova_senha = $_POST['nova_senha'] ?? '';

    if (!$senha_atual || !$nova_senha) {
        json_response($conexao, false, 'Erro: Preencha todos os campos');
    }

    // Verificar se a senha atual está correta
    $sql = "SELECT * FROM cadastro WHERE nome_usuario = '$usuario' AND senha = '$senha_atual'";
    $resultado = mysqli_query($conexao, $sql);

    if (mysqli_num_rows($resultado) === 0) {
        json_response($conexao, false, 'Erro: Senha atual incorreta');
    }

    // Atualizar a senha
    $sql_update = "UPDATE cadastro SET senha = '$nova_senha', conf_senha = '$nova_senha' WHERE nome_usuario = '$usuario'";
    $resultado_update = mysqli_query($conexao, $sql_update);

    if ($resultado_update) {
        json_response($conexao, true, 'Senha alterada com sucesso!');
    } else {
        json_response($conexao, false, 'Erro ao alterar senha: ' . mysqli_error($conexao));
    }
}

function alterarEmail($conexao, $usuario) {
    $novo_email = $_POST['novo_email'] ?? '';
    $senha = $_POST['senha'] ?? '';

    if (!$novo_email || !$senha) {
        json_response($conexao, false, 'Erro: Preencha todos os campos');
    }

    // Verificar se a senha está correta
    $sql = "SELECT * FROM cadastro WHERE nome_usuario = '$usuario' AND senha = '$senha'";
    $resultado = mysqli_query($conexao, $sql);

    if (mysqli_num_rows($resultado) === 0) {
        json_response($conexao, false, 'Erro: Senha incorreta');
    }

    // Verificar se o email já está em uso
    $sql_check = "SELECT * FROM cadastro WHERE email = '$novo_email' AND nome_usuario != '$usuario'";
    $resultado_check = mysqli_query($conexao, $sql_check);

    if (mysqli_num_rows($resultado_check) > 0) {
        json_response($conexao, false, 'Erro: Este email já está em uso');
    }

    // Atualizar o email
    $sql_update = "UPDATE cadastro SET email = '$novo_email' WHERE nome_usuario = '$usuario'";
    $resultado_update = mysqli_query($conexao, $sql_update);

    if ($resultado_update) {
        json_response($conexao, true, 'Email alterado com sucesso!');
    } else {
        json_response($conexao, false, 'Erro ao alterar email: ' . mysqli_error($conexao));
    }
}

function excluirConta($conexao, $usuario) {
    // Buscar o ID do usuário para excluir pelo ID
    $sql = "SELECT id FROM cadastro WHERE nome_usuario = '$usuario'";
    $resultado = mysqli_query($conexao, $sql);

    if (mysqli_num_rows($resultado) === 0) {
        json_response($conexao, false, 'Erro: Usuário não encontrado');
    }

    $dados = mysqli_fetch_assoc($resultado);
    $id_usuario = $dados['id'];

    // Excluir a conta pelo ID
    $sql_delete = "DELETE FROM cadastro WHERE id = '$id_usuario'";
    $resultado_delete = mysqli_query($conexao, $sql_delete);

    if ($resultado_delete) {
        json_response($conexao, true, 'Conta excluída com sucesso!');
    } else {
        json_response($conexao, false, 'Erro ao excluir conta: ' . mysqli_error($conexao));
    }
}
?>
