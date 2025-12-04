<?php
// Conexão
$conexao = mysqli_connect("localhost", "root", "", "login");

if (!$conexao) {
    die("Erro ao conectar ao banco: " . mysqli_connect_error());
}

// Receber ação
$acao = $_POST['acao'] ?? '';
$usuario = $_POST['usuario'] ?? '';

// Validar entrada
if (!$usuario) {
    die("Erro: Usuário não identificado");
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
        die("Erro: Preencha todos os campos");
    }

    // Verificar se a senha atual está correta
    $sql = "SELECT * FROM cadastro WHERE nome_usuario = '$usuario' AND senha = '$senha_atual'";
    $resultado = mysqli_query($conexao, $sql);

    if (mysqli_num_rows($resultado) === 0) {
        die("Erro: Senha atual incorreta");
    }

    // Atualizar a senha
    $sql_update = "UPDATE cadastro SET senha = '$nova_senha', conf_senha = '$nova_senha' WHERE nome_usuario = '$usuario'";
    $resultado_update = mysqli_query($conexao, $sql_update);

    if ($resultado_update) {
        echo "Senha alterada com sucesso!";
    } else {
        echo "Erro ao alterar senha: " . mysqli_error($conexao);
    }

    mysqli_close($conexao);
}

function alterarEmail($conexao, $usuario) {
    $novo_email = $_POST['novo_email'] ?? '';
    $senha = $_POST['senha'] ?? '';

    if (!$novo_email || !$senha) {
        die("Erro: Preencha todos os campos");
    }

    // Verificar se a senha está correta
    $sql = "SELECT * FROM cadastro WHERE nome_usuario = '$usuario' AND senha = '$senha'";
    $resultado = mysqli_query($conexao, $sql);

    if (mysqli_num_rows($resultado) === 0) {
        die("Erro: Senha incorreta");
    }

    // Verificar se o email já está em uso
    $sql_check = "SELECT * FROM cadastro WHERE email = '$novo_email' AND nome_usuario != '$usuario'";
    $resultado_check = mysqli_query($conexao, $sql_check);

    if (mysqli_num_rows($resultado_check) > 0) {
        die("Erro: Este email já está em uso");
    }

    // Atualizar o email
    $sql_update = "UPDATE cadastro SET email = '$novo_email' WHERE nome_usuario = '$usuario'";
    $resultado_update = mysqli_query($conexao, $sql_update);

    if ($resultado_update) {
        echo "Email alterado com sucesso!";
    } else {
        echo "Erro ao alterar email: " . mysqli_error($conexao);
    }

    mysqli_close($conexao);
}

function excluirConta($conexao, $usuario) {
    $senha = $_POST['senha'] ?? '';

    if (!$senha) {
        die("Erro: Senha necessária para confirmar exclusão");
    }

    // Verificar se a senha está correta
    $sql = "SELECT * FROM cadastro WHERE nome_usuario = '$usuario' AND senha = '$senha'";
    $resultado = mysqli_query($conexao, $sql);

    if (mysqli_num_rows($resultado) === 0) {
        die("Erro: Senha incorreta");
    }

    // Excluir a conta
    $sql_delete = "DELETE FROM cadastro WHERE nome_usuario = '$usuario'";
    $resultado_delete = mysqli_query($conexao, $sql_delete);

    if ($resultado_delete) {
        echo "Conta excluída com sucesso!";
    } else {
        echo "Erro ao excluir conta: " . mysqli_error($conexao);
    }

    mysqli_close($conexao);
}
?>
